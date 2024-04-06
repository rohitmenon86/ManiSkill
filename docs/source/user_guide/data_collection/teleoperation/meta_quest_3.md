# Meta Quest 3

To get started with using Meta Quest 3 for teleoperation in simulation and the real world, you need to go through a number of installation / setup steps

## Developer Account

You need to first create a developer Meta account in order to work with ALVR, an open source software for interfacing with applications.

To do so, follow the instructions at https://developer.oculus.com/documentation/native/android/mobile-device-setup/ to 
1. Create an organization for yourself
2. Verify your account (you need two-factor authentication **and** a registered phone number)

## Enabling Developer Mode on the Headset

To enable developer mode, at the moment you must have a mobile phone with Meta quest app installed. 

Once installed, login in to your developer account, and then follow the instructions to pair your headset with your phone. Note that your headset and your phone must be on the **same 5G wifi** in order to work for this section as well as future sections.

Once paired, go to Menu -> Devices -> Meta Quest 3 (the one you paired) -> Headset Settings -> Developer

Then enable developer mode

## Installing ALVR



## Running


Control Panda arm 
```bash
python mani_skill/examples/teleoperation/vr_panda.py -e "StackCube-v1"
```

Control fetch
```bash
python mani_skill/examples/teleoperation/vr_fetch.py -e "ReplicaCAD_SceneManipulation-v1"
```

Control two robot task setups
```bash
```