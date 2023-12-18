use std::collections::HashSet;
use anyhow::{anyhow, Error};
use vulkanalia::Entry;
use vulkanalia::vk::StringArray;
use crate::{VALIDATION_ENABLED, VALIDATION_LAYER};

pub unsafe fn create_layers(entry: &Entry) -> anyhow
{
    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but not supported."));
    }
    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };
}