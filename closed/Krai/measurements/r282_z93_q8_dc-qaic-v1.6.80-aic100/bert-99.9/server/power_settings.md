# Boot/BIOS Firmware Settings

## AMD CBS

### NBIO Common Options
#### SMU Common Options
##### Determinism Control: Manual
##### Determinism Slider: Power
##### cTDP Control: Auto
##### ABBDIS: Auto
##### DF Cstates: Auto

### DF Common Options
#### Scrubber
##### DRAM scrub time: Disabled
##### Poisson scrubber control: Disabled
##### Redirect scrubber control: Disabled

#### Memory Addressing
##### NUMA nodes per socket: NPS1
##### ACPI SRAT L3 Cche As NUMA Domain: Disabled

### CPU Common Options
#### Performance
##### SMT Control: Enable
#### Global C-state Control: Disabled

# Management Firmware Settings

Out-of-the-box.

# Power Management Settings

## Fan Speed (9,450 RPM)

<pre>
<b>&dollar;</b> sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>125</b> 0xFF
 0a 3c 00
</pre>

## Maximum Frequency

The maximum chip frequency is controlled through a variable called `vc`.
This variable is set automatically per workload and per system according to [cmdgen metadata](https://github.com/krai/ck-qaic/blob/MLC2.0/cmdgen/).
