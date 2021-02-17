# Import a general PID controller
from dsorlib.controllers.inner_loops.pid import PID

# Import surge (PI), yaw (PD) and yaw-rate (PI) controllers for an AUV and the corresponding outer-loops
from dsorlib.controllers.inner_loops.auv_inner_loops import yaw_rate_PI, yaw_PD, surge_PI
from dsorlib.controllers.outer_loops.auv_outer_loop import AUVOuterLoop

# Import the position and attitude controllers for a quadrotor
from dsorlib.controllers.inner_loops.quadrotor_inner_loops import QuadrotorInnerLoop

# Import the Path following virtual target implementations
from dsorlib.controllers.path_following.virtual_target import VirtualTarget, SimpleVirtualTarget