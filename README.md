# simple-bass-sim
Simulating various  Bass Guitar instruments in software

The software in this repo attempts to emulate certain Bass Guitar types. The idea stemmed from the Roland GK series of processors, and the research undertaken by Joel de Guzman in his cycfi project.

The idea takes a number of feed-forward comb filters to approximate the effect of a Bass Guitar's pickup and width, while also taking into account any comb filtering that will occur from the position of the GK pickup itself.

A simple reverb was added for the simulation of an acoustic Bass body, which is activated in preset #4.

The presets are approximations of the following:
1. Jazz Bass (two pickups, with the opportunity to blend between them)
2. P-Bass
3. Stingray
4. Acoustic Bass

Add PureData patch commit

After numerous weeks of 'chatting' with ChatGPT, I decided to start building a PureData patch by hand. Thanks to some great tutorials by QGCInteractiveMusic, SoundSimulator and Simon Hutchinson, I was able to get this patch to the state it is in now. This patch has not received any real-world testing as yet.
