use std::collections::HashSet;
use anyhow::{anyhow, Error};
use vulkanalia::Entry;
use vulkanalia::vk::StringArray;
use crate::{VALIDATION_ENABLED, VALIDATION_LAYER};
