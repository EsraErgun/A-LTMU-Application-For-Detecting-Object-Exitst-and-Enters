# A-LTMU-Application-For-Detecting-Object-Exitst-and-Enters
A PyTorch 1.7.1 Implementation of LTMU Meta Updater for Long Term Object Tracking. 
This is a Meta-updated LSTM implementation for long-term object tracking to detect enters and exists of an object to the scene. The Meta-LSTM is built as described in CVPR 2020 Paper "High-Performance Long-Term Tracking With Meta-Updater". This LSTM integrate geometric, discriminative, and appearance cues in a sequential manner, and then mine the sequential information with a designed cascaded LSTM module. The meta-updater learns a binary output to guide the tracker's update as explained in the original paper. 
