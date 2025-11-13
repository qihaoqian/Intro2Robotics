import numpy as np
import matplotlib.pyplot as plt
from problem5 import RRRArm, StiffnessController, create_3d_animation, plot_xy_trace

def run():
    arm = RRRArm()
    ctrl = StiffnessController(Kp=90.0, vel_limit=3.5)
    
    targets = np.array([[0.2,0.2,0.2],[0.1,0.2,0.2],[0.1,-0.2,0.2],[0.2,-0.2,0.2],[0.2,0.2,0.2]])
    
    T = 5.0
    dt = 0.01
    N = int(T/dt)
    move_steps = 100
    dwell_steps = 20
    
    sequence = []
    for leg in range(4):
        p0, p1 = targets[leg], targets[leg+1]
        for k in range(move_steps):
            s = (k+1)/move_steps
            sequence.append((1-s)*p0 + s*p1)
        for k in range(dwell_steps):
            sequence.append(p1.copy())
    
    while len(sequence) < N:
        sequence.append(targets[-1].copy())
    sequence = np.array(sequence[:N])
    
    q = arm.ik(targets[0])
    x_prev, _ = arm.fk(q)
    x_log = []
    frames = []
    
    for i in range(N):
        q, x = ctrl.step(arm, q, sequence[i], x_prev, dt)
        x_prev = x.copy()
        x_log.append(x.copy())
        _, joints = arm.fk(q)
        frames.append(np.stack(joints))
    
    x_log = np.array(x_log)
    frames = np.array(frames)
    
    create_3d_animation(frames, dt, targets, sequence, "results/problem6_2_result.gif", N_per=None)
    
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(N)*dt, x_log[:,2])
    plt.axhline(0.2, linestyle='--', color='gray', alpha=0.7)
    plt.title("Z with waypoint dwell")
    plt.xlabel("time (s)")
    plt.ylabel("Z (m)")
    plt.savefig("results/problem6_2_z_time.png", dpi=140, bbox_inches='tight')
    plt.close()
    
    plot_xy_trace(x_log, targets, "results/problem6_2_xy_trace.png", 
                  title="XY with waypoint dwell")
    
    print("problem 6 gif saved to results/problem6_2_result.gif")
    print("results saved to results/problem6_2_z_time.png and results/problem6_2_xy_trace.png")

if __name__ == "__main__":
    run()
