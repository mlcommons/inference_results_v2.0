# Boot/BIOS Firmware Settings

Out-of-the-box.

# Management Firmware Settings

Out-of-the-box.

# Power Management Settings

## TDP Settings

### Set 25W TDP

<pre>
<b>[anton@aedk3 ~]&dollar;</b> echo 25000000 | sudo tee /sys/class/hwmon/hwmon*/power1_max
25000000
</pre>

### Reboot

<pre>
<b>[anton@aedk3 ~]&dollar;</b> sudo reboot
</pre>

### Check 25W TDP

<pre>
<b>[anton@aedk3 ~]&dollar;</b> cat /sys/class/hwmon/hwmon*/power1_max
25000000
</pre>
