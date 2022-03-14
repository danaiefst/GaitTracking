# GaitTracking

This is the repository for the code of paper [Deep Leg Tracking by Detection and Gait Analysis in 2D Range Data for Intelligent Robotic Assistants](https://doi.org/10.1109/IROS51168.2021.9636588).

The database used for training the network can be found [here](https://robotics.ntua.gr/ltgad/).

On main branch exists the Leg Tracking network training code
On gait branch exists the Gait Analysis network training code

File decription:
* model.pt, best_model.pt: PyTorch pretrained models
* data_handler.py: Custom DataLoader that turns laserpoints into occupancy grids and also loads center and gait state annotations
* check_batches.py: An example usage of the data_handler.py for visual representation of the data
* tracking_nn.py: The network architecture
* train.py: The training and validation code
* test.py: The final testing code

File paths should be changed in order for the code to run.
