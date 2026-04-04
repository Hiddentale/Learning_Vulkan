use anyhow::Context;
use glam::{Mat4, Quat, Vec3};
use log::{info, warn};
use openxr as xr;
use std::ffi::CString;
use std::os::raw::c_void;
use vk::Handle;
use vulkan_rust::vk;

/// Result of probing the system for VR support.
pub enum VrSupport {
    /// OpenXR runtime found, HMD available, Vulkan requirements met.
    Available(VrContext),
    /// No runtime or no HMD — desktop-only fallback.
    Unavailable(String),
}

/// Holds OpenXR state needed to create Vulkan objects and later a session.
///
/// Lifecycle: created early (before Vulkan instance), consumed when
/// the OpenXR session is started.
pub struct VrContext {
    pub xr_instance: xr::Instance,
    pub system: xr::SystemId,
    pub requirements: xr::vulkan::Requirements,
    /// Whether the legacy `khr_vulkan_enable` extension is available for
    /// querying required Vulkan extensions.
    pub has_legacy_queries: bool,
}

/// Active OpenXR session bound to a Vulkan device.
pub struct VrSession {
    pub xr_instance: xr::Instance,
    system: xr::SystemId,
    pub session: xr::Session<xr::Vulkan>,
    pub frame_waiter: xr::FrameWaiter,
    pub frame_stream: xr::FrameStream<xr::Vulkan>,
    pub stage_space: xr::Space,
    state: xr::SessionState,
    event_buffer: xr::EventDataBuffer,
}

impl VrSession {
    pub fn system(&self) -> xr::SystemId {
        self.system
    }

    /// Whether the session is in a state where frames should be rendered.
    pub fn is_running(&self) -> bool {
        matches!(
            self.state,
            xr::SessionState::SYNCHRONIZED | xr::SessionState::VISIBLE | xr::SessionState::FOCUSED
        )
    }

    /// Poll all pending OpenXR events and handle session state transitions.
    /// Returns `false` if the session should be abandoned (EXITING or LOSS_PENDING).
    pub fn poll_events(&mut self) -> anyhow::Result<bool> {
        loop {
            let event = self.xr_instance.poll_event(&mut self.event_buffer)?;
            let Some(event) = event else { return Ok(true) };

            match event {
                xr::Event::SessionStateChanged(changed) => {
                    let new_state = changed.state();
                    info!("OpenXR session state: {:?} → {:?}", self.state, new_state);
                    self.state = new_state;

                    match new_state {
                        xr::SessionState::READY => {
                            self.session
                                .begin(xr::ViewConfigurationType::PRIMARY_STEREO)
                                .context("failed to begin session")?;
                            info!("OpenXR session begun");
                        }
                        xr::SessionState::STOPPING => {
                            self.session.end().context("failed to end session")?;
                            info!("OpenXR session ended");
                        }
                        xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                            return Ok(false);
                        }
                        _ => {}
                    }
                }
                xr::Event::InstanceLossPending(_) => {
                    warn!("OpenXR instance loss pending");
                    return Ok(false);
                }
                _ => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// OpenXR pose → glam matrix conversions
// ---------------------------------------------------------------------------

const NEAR_PLANE: f32 = 0.1;
const FAR_PLANE: f32 = 500.0;

/// Convert an OpenXR View (pose + FOV) to a view-projection matrix.
pub fn view_to_vp(view: &xr::View) -> Mat4 {
    let proj = projection_from_fov(&view.fov);
    let view_mat = view_matrix_from_pose(&view.pose);
    proj * view_mat
}

/// Build a Vulkan-compatible projection matrix from OpenXR asymmetric FOV.
fn projection_from_fov(fov: &xr::Fovf) -> Mat4 {
    let left = fov.angle_left.tan();
    let right = fov.angle_right.tan();
    let up = fov.angle_up.tan();
    let down = fov.angle_down.tan();

    let width = right - left;
    let height = up - down;

    // Reversed-depth or standard [0,1] Vulkan depth, right-handed
    let mut m = Mat4::ZERO;
    m.x_axis.x = 2.0 / width;
    m.y_axis.y = -2.0 / height; // Vulkan Y-down
    m.z_axis.x = (right + left) / width;
    m.z_axis.y = -(up + down) / height; // Vulkan Y-down
    m.z_axis.z = -FAR_PLANE / (FAR_PLANE - NEAR_PLANE);
    m.z_axis.w = -1.0;
    m.w_axis.z = -(FAR_PLANE * NEAR_PLANE) / (FAR_PLANE - NEAR_PLANE);
    m
}

/// Convert an OpenXR pose (position + orientation) to a view matrix.
fn view_matrix_from_pose(pose: &xr::Posef) -> Mat4 {
    let q = &pose.orientation;
    let p = &pose.position;
    let rotation = Quat::from_xyzw(q.x, q.y, q.z, q.w);
    let translation = Vec3::new(p.x, p.y, p.z);
    Mat4::from_rotation_translation(rotation, translation).inverse()
}

impl VrContext {
    /// Attempt to load the OpenXR runtime, find an HMD, and query Vulkan
    /// graphics requirements.  Returns `VrSupport::Unavailable` with a
    /// human-readable reason if any step fails non-fatally.
    pub fn probe() -> anyhow::Result<VrSupport> {
        let entry = match load_entry() {
            Ok(e) => e,
            Err(reason) => return Ok(VrSupport::Unavailable(reason)),
        };

        let extensions = match entry.enumerate_extensions() {
            Ok(ext) => ext,
            Err(e) => return Ok(VrSupport::Unavailable(format!("cannot enumerate extensions: {e}"))),
        };

        if !extensions.khr_vulkan_enable2 {
            return Ok(VrSupport::Unavailable("runtime does not support XR_KHR_vulkan_enable2".into()));
        }

        let has_legacy_queries = extensions.khr_vulkan_enable;
        let xr_instance = create_xr_instance(&entry, &extensions)?;
        let system = match find_hmd(&xr_instance) {
            Ok(s) => s,
            Err(reason) => return Ok(VrSupport::Unavailable(reason)),
        };

        let requirements = xr_instance
            .graphics_requirements::<xr::Vulkan>(system)
            .context("failed to query Vulkan graphics requirements")?;

        info!(
            "OpenXR Vulkan requirements: min={}, max={}",
            requirements.min_api_version_supported, requirements.max_api_version_supported
        );

        Ok(VrSupport::Available(VrContext {
            xr_instance,
            system,
            requirements,
            has_legacy_queries,
        }))
    }

    /// Query Vulkan instance extensions required by the OpenXR runtime.
    pub fn required_instance_extensions(&self) -> Vec<CString> {
        if !self.has_legacy_queries {
            warn!("khr_vulkan_enable not available — cannot query required instance extensions");
            return Vec::new();
        }
        match self.xr_instance.vulkan_legacy_instance_extensions(self.system) {
            Ok(ext_string) => parse_extension_string(&ext_string),
            Err(e) => {
                warn!("Failed to query VR instance extensions: {e}");
                Vec::new()
            }
        }
    }

    /// Query Vulkan device extensions required by the OpenXR runtime.
    pub fn required_device_extensions(&self) -> Vec<CString> {
        if !self.has_legacy_queries {
            warn!("khr_vulkan_enable not available — cannot query required device extensions");
            return Vec::new();
        }
        match self.xr_instance.vulkan_legacy_device_extensions(self.system) {
            Ok(ext_string) => parse_extension_string(&ext_string),
            Err(e) => {
                warn!("Failed to query VR device extensions: {e}");
                Vec::new()
            }
        }
    }

    /// Ask the OpenXR runtime which physical device it prefers.
    ///
    /// # Safety
    /// `vk_instance` must be a valid Vulkan instance.
    pub unsafe fn preferred_gpu(&self, vk_instance: vk::Instance) -> anyhow::Result<vk::PhysicalDevice> {
        let xr_vk_instance = vk_handle_to_xr(vk_instance);
        let raw_device = self
            .xr_instance
            .vulkan_graphics_device(self.system, xr_vk_instance)
            .context("failed to query VR preferred GPU")?;
        Ok(xr_ptr_to_vk_handle(raw_device))
    }

    /// Create an OpenXR session bound to the given Vulkan device.
    ///
    /// Consumes the `VrContext` and returns a `VrSession` ready for the
    /// frame loop.
    ///
    /// # Safety
    /// All Vulkan handles must be valid and the queue must support graphics.
    pub unsafe fn create_session(
        self,
        vk_instance: vk::Instance,
        vk_physical_device: vk::PhysicalDevice,
        vk_device: vk::Device,
        queue_family_index: u32,
    ) -> anyhow::Result<VrSession> {
        let session_create_info = xr::vulkan::SessionCreateInfo {
            instance: vk_handle_to_xr(vk_instance),
            physical_device: vk_handle_to_xr(vk_physical_device),
            device: vk_handle_to_xr(vk_device),
            queue_family_index,
            queue_index: 0,
        };

        let (session, frame_waiter, frame_stream) = self
            .xr_instance
            .create_session::<xr::Vulkan>(self.system, &session_create_info)
            .context("failed to create OpenXR session")?;

        let stage_space = session
            .create_reference_space(xr::ReferenceSpaceType::STAGE, xr::Posef::IDENTITY)
            .context("failed to create STAGE reference space")?;

        info!("OpenXR session created successfully");

        Ok(VrSession {
            xr_instance: self.xr_instance,
            system: self.system,
            session,
            frame_waiter,
            frame_stream,
            stage_space,
            state: xr::SessionState::IDLE,
            event_buffer: xr::EventDataBuffer::default(),
        })
    }
}

// ---------------------------------------------------------------------------
// Handle conversion utilities
// ---------------------------------------------------------------------------

/// Convert a vulkan-rs dispatchable handle to the `*const c_void` that
/// openxr-sys expects.
pub fn vk_handle_to_xr<H: Handle<Repr = usize>>(handle: H) -> *const c_void {
    handle.as_raw() as *const c_void
}

/// Convert an openxr-sys `*const c_void` back to a vulkan-rs dispatchable
/// handle.
///
/// # Safety
/// The pointer must have originated from a valid Vulkan dispatchable handle.
pub unsafe fn xr_ptr_to_vk_handle<H: Handle<Repr = usize>>(ptr: *const c_void) -> H {
    H::from_raw(ptr as usize)
}

/// Convert a vulkan-rs `vk::Image` (non-dispatchable, `u64`) to the `u64`
/// that openxr returns for swapchain images.
pub fn vk_image_to_xr(image: vk::Image) -> u64 {
    image.as_raw()
}

/// Convert an openxr swapchain image handle (`u64`) to a vulkan-rs `vk::Image`.
pub(crate) fn xr_image_to_vk(handle: u64) -> vk::Image {
    vk::Image::from_raw(handle)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn load_entry() -> Result<xr::Entry, String> {
    match unsafe { xr::Entry::load() } {
        Ok(entry) => {
            info!("OpenXR runtime loaded successfully");
            Ok(entry)
        }
        Err(e) => {
            warn!("No OpenXR runtime found: {e}");
            Err(format!("no OpenXR runtime: {e}"))
        }
    }
}

/// Parse a space-delimited extension string into a list of CStrings.
fn parse_extension_string(extensions: &str) -> Vec<CString> {
    extensions
        .split_whitespace()
        .filter(|s| !s.is_empty())
        .filter_map(|s| CString::new(s).map_err(|e| warn!("Skipping malformed extension name: {e}")).ok())
        .collect()
}

fn create_xr_instance(entry: &xr::Entry, available: &xr::ExtensionSet) -> anyhow::Result<xr::Instance> {
    let mut required = xr::ExtensionSet::default();
    required.khr_vulkan_enable2 = true;

    // Enable legacy Vulkan extension for querying required Vulkan extensions.
    if available.khr_vulkan_enable {
        required.khr_vulkan_enable = true;
    }

    // Opt in to depth layer submission if available (useful for reprojection).
    if available.khr_composition_layer_depth {
        required.khr_composition_layer_depth = true;
    }

    let instance = entry
        .create_instance(
            &xr::ApplicationInfo {
                application_name: "Voxel Engine",
                application_version: 1,
                engine_name: "Custom",
                engine_version: 1,
                api_version: xr::Version::new(1, 0, 0),
            },
            &required,
            &[],
        )
        .context("failed to create OpenXR instance")?;

    let props = instance.properties().context("failed to query instance properties")?;
    info!("OpenXR runtime: {} v{}", props.runtime_name, props.runtime_version);

    Ok(instance)
}

fn find_hmd(instance: &xr::Instance) -> Result<xr::SystemId, String> {
    match instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY) {
        Ok(system) => {
            info!("HMD system found: {:?}", system);
            Ok(system)
        }
        Err(xr::sys::Result::ERROR_FORM_FACTOR_UNAVAILABLE) => {
            warn!("No HMD connected");
            Err("no HMD connected".into())
        }
        Err(e) => Err(format!("failed to find HMD system: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Handle conversion round-trip tests ----------------------------------

    #[test]
    fn dispatchable_handle_roundtrip() {
        // Simulate a non-null VkInstance handle
        let original = vk::Instance::from_raw(0xDEAD_BEEF);
        let xr_ptr = vk_handle_to_xr(original);
        let recovered: vk::Instance = unsafe { xr_ptr_to_vk_handle(xr_ptr) };
        assert_eq!(original, recovered);
    }

    #[test]
    fn physical_device_handle_roundtrip() {
        let original = vk::PhysicalDevice::from_raw(0x1234_5678);
        let xr_ptr = vk_handle_to_xr(original);
        let recovered: vk::PhysicalDevice = unsafe { xr_ptr_to_vk_handle(xr_ptr) };
        assert_eq!(original, recovered);
    }

    #[test]
    fn device_handle_roundtrip() {
        let original = vk::Device::from_raw(0xCAFE_BABE);
        let xr_ptr = vk_handle_to_xr(original);
        let recovered: vk::Device = unsafe { xr_ptr_to_vk_handle(xr_ptr) };
        assert_eq!(original, recovered);
    }

    #[test]
    fn null_handle_roundtrip() {
        let null = vk::Instance::null();
        let xr_ptr = vk_handle_to_xr(null);
        assert!(xr_ptr.is_null());
        let recovered: vk::Instance = unsafe { xr_ptr_to_vk_handle(xr_ptr) };
        assert!(recovered.is_null());
    }

    // -- Non-dispatchable (VkImage) conversion tests -------------------------

    #[test]
    fn image_handle_roundtrip() {
        let original = vk::Image::from_raw(0x0000_FFFF_DEAD_BEEF);
        let xr_val = vk_image_to_xr(original);
        let recovered = xr_image_to_vk(xr_val);
        assert_eq!(original, recovered);
    }

    #[test]
    fn null_image_roundtrip() {
        let null = vk::Image::null();
        let xr_val = vk_image_to_xr(null);
        assert_eq!(xr_val, 0);
        let recovered = xr_image_to_vk(xr_val);
        assert!(recovered.is_null());
    }

    // -- Size/alignment sanity checks ----------------------------------------

    #[test]
    fn dispatchable_handle_is_pointer_sized() {
        assert_eq!(
            std::mem::size_of::<vk::Instance>(),
            std::mem::size_of::<*const c_void>(),
            "VkInstance must be pointer-sized for safe cast to *const c_void"
        );
    }

    #[test]
    fn non_dispatchable_handle_is_u64() {
        assert_eq!(
            std::mem::size_of::<vk::Image>(),
            std::mem::size_of::<u64>(),
            "VkImage must be u64-sized to match openxr-sys SwapchainImage"
        );
    }

    #[test]
    fn xr_vulkan_session_create_info_layout() {
        // Verify the SessionCreateInfo struct we'll construct matches expected fields.
        let info = xr::vulkan::SessionCreateInfo {
            instance: std::ptr::null(),
            physical_device: std::ptr::null(),
            device: std::ptr::null(),
            queue_family_index: 42,
            queue_index: 0,
        };
        assert_eq!(info.queue_family_index, 42);
        assert_eq!(info.queue_index, 0);
    }
}
