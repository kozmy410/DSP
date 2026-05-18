import sounddevice as sd
import json
import os

def get_host_apis():
    """
    Retrieves all available audio host APIs on the system.
    Returns a dictionary mapping the API index to its details.
    """
    apis = sd.query_hostapis()
    api_dict = {}
    for idx, api in enumerate(apis):
        api_dict[idx] = {
            "name": api.get("name"),
            "default_input_device": api.get("default_input_device"),
            "default_output_device": api.get("default_output_device")
        }
    return api_dict

def get_all_devices():
    """
    Retrieves all audio devices, categorizing them into inputs and outputs.
    """
    devices = sd.query_devices()
    apis = get_host_apis()
    
    input_devices = []
    output_devices = []
    
    for idx, dev in enumerate(devices):
        api_index = dev.get("hostapi")
        api_name = apis.get(api_index, {}).get("name", "Unknown API")
        
        device_info = {
            "index": idx,
            "name": dev.get("name"),
            "host_api": api_name,
            "max_inputs": dev.get("max_input_channels"),
            "max_outputs": dev.get("max_output_channels"),
            "default_samplerate": dev.get("default_samplerate")
        }
        
        if device_info["max_inputs"] > 0:
            input_devices.append(device_info)
        if device_info["max_outputs"] > 0:
            output_devices.append(device_info)
            
    return {"inputs": input_devices, "outputs": output_devices}

def probe_sample_rates(device_index, is_input=True):
    """
    Tests common sample rates to see which ones the specified device supports natively.
    """
    standard_rates = [44100, 48000, 88200, 96000, 192000]
    supported_rates = []
    
    channels = sd.query_devices(device_index).get("max_input_channels" if is_input else "max_output_channels")
    if channels == 0:
        return supported_rates

    for rate in standard_rates:
        try:
            if is_input:
                sd.check_input_settings(device=device_index, channels=channels, samplerate=rate)
            else:
                sd.check_output_settings(device=device_index, channels=channels, samplerate=rate)
            supported_rates.append(rate)
        except Exception:
            pass
            
    return supported_rates

def generate_hardware_config(output_filename="hardware_config.json"):
    """
    Sweeps the system, probes capabilities, and saves a configuration profile.
    This file can be read by the GUI to populate dropdown menus.
    """
    print("Initiating hardware sweep...")
    config_data = {
        "host_apis": get_host_apis(),
        "devices": get_all_devices(),
        "system_defaults": {}
    }
    
    try:
        default_in, default_out = sd.default.device
        config_data["system_defaults"]["input_device_index"] = default_in
        config_data["system_defaults"]["output_device_index"] = default_out
    except Exception:
        config_data["system_defaults"]["input_device_index"] = -1
        config_data["system_defaults"]["output_device_index"] = -1

    # Probe supported sample rates for all input devices to ensure safe recording
    print("Probing input device capabilities...")
    for dev in config_data["devices"]["inputs"]:
        dev["supported_rates"] = probe_sample_rates(dev["index"], is_input=True)

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)
        
    print(f"Hardware configuration saved to: {os.path.abspath(output_filename)}")
    return config_data

def load_hardware_config(input_filename="hardware_config.json"):
    """
    Loads the hardware configuration profile.
    """
    if not os.path.exists(input_filename):
        print("Hardware config not found. Generating a new one...")
        return generate_hardware_config(input_filename)
        
    with open(input_filename, "r", encoding="utf-8") as f:
        return json.load(f)