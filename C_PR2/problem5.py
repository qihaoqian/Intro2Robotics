
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class RRRArm:
    l1: float = 0.05
    l2: float = 0.20
    l3: float = 0.15
    
    def fk(self, q: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        q1, q2, q3 = q
        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)
        c23, s23 = np.cos(q2+q3), np.sin(q2+q3)
        O0 = np.array([0.0,0.0,0.0])
        O1 = np.array([0.0,0.0,self.l1])
        r = self.l2*c2 + self.l3*c23
        x = r*c1
        y = r*s1
        z = self.l1 + self.l2*s2 + self.l3*s23
        ee = np.array([x,y,z])
        elbow = np.array([self.l2*c2*c1, self.l2*c2*s1, self.l1 + self.l2*s2])
        wrist = ee
        return ee, [O0,O1,elbow,wrist]
    
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        q1, q2, q3 = q
        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)
        c23, s23 = np.cos(q2+q3), np.sin(q2+q3)
        r = self.l2*c2 + self.l3*c23
        J = np.zeros((3,3))
        J[0,0] = -r*s1; J[1,0] = r*c1; J[2,0] = 0.0
        drdq2 = -self.l2*s2 - self.l3*s23
        J[0,1] = drdq2*c1; J[1,1] = drdq2*s1; J[2,1] = r
        drdq3 = -self.l3*s23
        J[0,2] = drdq3*c1; J[1,2] = drdq3*s1; J[2,2] = self.l3*c23
        return J
    
    def ik(self, xyz: np.ndarray, elbow_up: bool=False) -> np.ndarray:
        x,y,z = xyz; q1 = np.arctan2(y,x); rxy = np.hypot(x,y); zp = z - self.l1
        L2,L3 = self.l2, self.l3
        D = (rxy**2 + zp**2 - L2**2 - L3**2)/(2*L2*L3); D = np.clip(D,-1,1)
        if elbow_up: q3 = np.arctan2(+np.sqrt(max(0.0,1-D**2)), D)
        else:        q3 = np.arctan2(-np.sqrt(max(0.0,1-D**2)), D)
        q2 = np.arctan2(zp, rxy) - np.arctan2(L3*np.sin(q3), L2 + L3*np.cos(q3))
        return np.array([q1,q2,q3])

@dataclass
class StiffnessController:
    Kp: float = 80.0
    zeta: float = 1.0
    mass_virtual: float = 1.0
    vel_limit: float = 4.0
    damping_pinv: float = 1e-3
    
    def gains(self):
        wn = (self.Kp/self.mass_virtual)**0.5
        return self.Kp, 2*self.zeta*wn*self.mass_virtual
    
    def step(self, arm: RRRArm, q: np.ndarray, xd: np.ndarray, x_prev: np.ndarray, dt: float) -> tuple:
        x,_ = arm.fk(q); J = arm.jacobian(q); xdot = (x - x_prev)/dt
        Kp, Kd = self.gains(); F = Kp*(xd - x) - Kd*xdot
        JT = J.T; JJt = J@JT; lam = self.damping_pinv
        qdot = JT @ np.linalg.inv(JJt + (lam**2)*np.eye(3)) @ F
        nrm = np.linalg.norm(qdot); 
        if nrm > self.vel_limit: qdot = qdot*(self.vel_limit/max(nrm,1e-9))
        return q + qdot*dt, x

def create_3d_animation(frames, dt, targets, sequence, output_path, N_per=None):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-0.35,0.35)
    ax.set_ylim(-0.35,0.35)
    ax.set_zlim(0.0,0.45)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    line, = ax.plot([], [], [], lw=3, marker='o')
    goal_scatter = ax.scatter([], [], [], s=30, c='red')
    th = np.linspace(0,2*np.pi,60)
    r = 0.05
    ax.plot(r*np.cos(th), r*np.sin(th), 0*th, lw=1, color='gray')
    
    def init():
        line.set_data([],[])
        line.set_3d_properties([])
        return line,
    
    def update(i):
        pts = frames[i]
        xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]
        line.set_data(xs, ys)
        line.set_3d_properties(zs)
        if N_per is not None:
            seg = min(i // N_per, len(targets)-2)
            goal = targets[seg+1]
        else:
            goal = sequence[i]
        goal_scatter._offsets3d = ([goal[0]], [goal[1]], [goal[2]])
        return line,
    
    anim = FuncAnimation(fig, update, init_func=init, frames=len(frames), 
                        interval=1000*dt, blit=True)
    anim.save(output_path, writer=PillowWriter(fps=int(1/dt)))
    plt.close(fig)

def plot_xy_trace(x_log, targets, output_path, title="End-effector XY Trace"):
    plt.figure(figsize=(5,5))
    plt.plot(x_log[:,0], x_log[:,1], linewidth=2, label="trajectory")
    plt.plot(targets[:,0], targets[:,1], 'o--', label="waypoints")
    plt.axis('equal')
    plt.legend()
    plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def run():
    arm = RRRArm()
    ctrl = StiffnessController()
    targets = np.array([[0.2,0.2,0.2],[0.1,0.2,0.2],[0.1,-0.2,0.2],[0.2,-0.2,0.2],[0.2,0.2,0.2]])
    q = arm.ik(targets[0])
    dt = 0.02
    segment_time = 1.2
    N_per = int(segment_time/dt)
    all_x = []
    frames = []
    x_prev,_ = arm.fk(q)
    
    for seg in range(len(targets)-1):
        xd = targets[seg+1]
        for _ in range(N_per):
            q, x = ctrl.step(arm, q, xd, x_prev, dt)
            x_prev = x.copy()
            all_x.append(x.copy())
            _, joints = arm.fk(q)
            frames.append(np.stack(joints))
    
    frames = np.array(frames)
    all_x = np.array(all_x)
    
    create_3d_animation(frames, dt, targets, None, "results/problem5_result.gif", N_per)
    plot_xy_trace(all_x, targets, "results/problem5_result.png")
    
    print("problem 5 gif saved to results/problem5_result.gif")
    print("results saved to results/problem5_result.png")

if __name__ == "__main__":
    run()
