import numpy as np
import time
import os
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
import Planner
import Planner_weighted_Astar
import Planner_bi_RRT

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))


def load_map(fname):
  '''
  Loads the bounady and blocks from map file fname.

  boundary = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]

  blocks = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'],
            ...,
            ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
  '''
  mapdata = np.loadtxt(fname,dtype={'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'),\
                                    'formats': ('S8','f', 'f', 'f', 'f', 'f', 'f', 'f','f','f')})
  blockIdx = mapdata['type'] == b'block'
  boundary = mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
  blocks = mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
  return boundary, blocks


def draw_map(boundary, blocks, start, goal):
  '''
  Visualization of a planning problem with environment boundary, obstacle blocks, and start and goal points
  '''
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  hb = draw_block_list(ax,blocks)
  hs = ax.plot(start[0:1],start[1:2],start[2:],'ro',markersize=7,markeredgecolor='k')
  hg = ax.plot(goal[0:1],goal[1:2],goal[2:],'go',markersize=7,markeredgecolor='k')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim(boundary[0,0],boundary[0,3])
  ax.set_ylim(boundary[0,1],boundary[0,4])
  ax.set_zlim(boundary[0,2],boundary[0,5])
  return fig, ax, hb, hs, hg

def draw_block_list(ax,blocks):
  '''
  Subroutine used by draw_map() to display the environment blocks
  '''
  v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype='float')
  f = np.array([[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[0,1,2,3],[4,5,6,7]])
  clr = blocks[:,6:]/255
  n = blocks.shape[0]
  d = blocks[:,3:6] - blocks[:,:3]
  vl = np.zeros((8*n,3))
  fl = np.zeros((6*n,4),dtype='int64')
  fcl = np.zeros((6*n,3))
  for k in range(n):
    vl[k*8:(k+1)*8,:] = v * d[k] + blocks[k,:3]
    fl[k*6:(k+1)*6,:] = f + k*8
    fcl[k*6:(k+1)*6,:] = clr[k,:]

  if type(ax) is Poly3DCollection:
    ax.set_verts(vl[fl])
  else:
    pc = Poly3DCollection(vl[fl], alpha=0.25, linewidths=1, edgecolors='k')
    pc.set_facecolor(fcl)
    h = ax.add_collection3d(pc)
    return h

def plot_performance(planner, weight, labels, lengths, times, verbose=False, save_dir=None):
    x = np.arange(len(labels))
    width = 0.4

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = 'C0'
    bars1 = ax1.bar(x - width/2, lengths, width, label='Path Length', color=color1)
    ax1.set_ylabel('Path Length', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, max(lengths)*1.1)

    ax2 = ax1.twinx()
    color2 = 'C1'
    bars2 = ax2.bar(x + width/2, times, width, label='Planning Time (s)', color=color2)
    ax2.set_ylabel('Planning Time (s)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(times)*1.1)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    if planner == 'bi_RRT':
      ax1.set_title('Comparison of Path Length and Planning Time for Bi-RRT Planner')
    elif planner == 'weighted_Astar':
      ax1.set_title(f'Comparison of Path Length and Planning Time for Weighted A* Planner (weight={weight})')

    ax1.bar_label(bars1, fmt='%.2f', padding=3, color=color1)
    ax2.bar_label(bars2, fmt='%.2f', padding=3, color=color2)
    ax1.legend([bars1, bars2], [bars1.get_label(), bars2.get_label()], loc='upper left')

    plt.tight_layout()

    if save_dir:
        out = os.path.join(save_dir, 'performance.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  saved: {out}")

    if verbose:
        plt.show(block=True)
    else:
        plt.close(fig)

def runtest(planner, mapfile, start, goal, verbose=False, weight=1.0, res=0.5, save_dir=None):
  '''
  This function:
   * loads the provided mapfile
   * creates a motion planner
   * plans a path from start to goal
   * checks whether the path is collision free and reaches the goal
   * computes the path length as a sum of the Euclidean norm of the path segments
  '''
  boundary, blocks = load_map(mapfile)
  if planner == 'bi_RRT':
    MP = Planner_bi_RRT.MyPlanner(boundary=boundary, blocks=blocks, res=res)
  elif planner == 'weighted_Astar':
    MP = Planner_weighted_Astar.MyPlanner(boundary=boundary, blocks=blocks, res=res, weight=weight)
  else:
    MP = Planner.MyPlanner(boundary=boundary, blocks=blocks, res=res)

  need_fig = verbose or (save_dir is not None)
  if need_fig:
    fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)
    map_name = os.path.splitext(os.path.basename(mapfile))[0]
    fig.suptitle(map_name, fontsize=12)

  t0 = tic()
  path = MP.plan(start=start, goal=goal)
  delta_time = time.time() - t0

  if path is not None:
    success = True
    pathlength = np.sum(np.sqrt(np.sum(np.diff(path,axis=0)**2,axis=1)))
    if need_fig:
      ax.plot(path[:,0], path[:,1], path[:,2], 'r-')
  else:
    success = False
    pathlength = 0.0

  if need_fig:
    if save_dir:
      out = os.path.join(save_dir, f"{map_name}_path.png")
      fig.savefig(out, dpi=150, bbox_inches='tight')
      print(f"  saved: {out}")
    if not verbose:
      plt.close(fig)

  return success, pathlength, delta_time

if __name__=="__main__":
  parser = argparse.ArgumentParser(description='Test the motion planner.')
  parser.add_argument('--planner', type=str, default='bi_RRT',
                      help='The motion planner to use (default: bi_RRT), choices: [bi_RRT, weighted_Astar]')
  parser.add_argument('--weight', type=float, default=1.0,
                      help='The weight for the weighted A* planner (default: 1.0)')
  parser.add_argument('--verbose', action='store_true',
                      help='Display interactive plots')
  parser.add_argument('--save', type=str, default=None, metavar='DIR',
                      help='Save stats (CSV) and figures (PNG) to DIR/<planner>_<timestamp>/')
  args = parser.parse_args()

  # Configure matplotlib backend before any figure is created.
  # Use non-interactive Agg when only saving (no display required).
  if args.verbose:
    plt.ion()
  else:
    plt.switch_backend('Agg')

  planner = args.planner
  weight  = args.weight
  verbose = args.verbose

  # Prepare output directory if --save is requested
  save_dir = None
  if args.save:
    run_tag  = f"{planner}_w{weight}" if planner == 'weighted_Astar' else planner
    run_dir  = os.path.join(args.save, run_tag)
    os.makedirs(run_dir, exist_ok=True)
    save_dir = run_dir
    print(f"Results will be saved to: {save_dir}")

  tests = [
      ("single_cube",    "./maps/single_cube.txt",    np.array([7.0,7.0,5.5]),  np.array([2.3,2.3,1.3]),  0.5),
      ("maze",           "./maps/maze.txt",           np.array([0.0,0.0,1.0]),  np.array([12.0,12.0,5.0]),  0.5),
      ("flappy_bird",    "./maps/flappy_bird.txt",    np.array([0.5,4.5,5.5]),  np.array([19.5,1.5,1.5]),  0.5),
      ("pillars",        "./maps/pillars.txt",        np.array([0.5,0.5,0.5]),  np.array([19.0,19.0,9.0]), 0.5),
      ("window",         "./maps/window.txt",         np.array([6.0,-4.9,2.8]), np.array([2.0,19.5,5.5]),  0.5),
      ("tower",          "./maps/tower.txt",          np.array([4.0,2.5,19.5]), np.array([2.5,4.0,0.5]),  0.5),
      ("room",           "./maps/room.txt",           np.array([1.0,5.0,1.5]),  np.array([9.0,7.0,1.5]), 0.5),
  ]

  labels  = []
  lengths = []
  times   = []
  results = []   # (name, success, path_length, planning_time) for all tests

  for name, fname, start, goal, res in tests:
      success, L, T = runtest(planner, fname, start, goal,
                               res=res, weight=weight, verbose=verbose,
                               save_dir=save_dir)
      results.append((name, success, L, T))
      if success:
        labels.append(name)
        lengths.append(L)
        times.append(T)
        print(f"Test {name:12s}  PASS  length: {L:.3f}  time: {T:.3f}s")
      else:
        print(f"Test {name:12s}  FAIL")

  # Save stats CSV
  if save_dir:
    csv_path = os.path.join(save_dir, 'stats.csv')
    with open(csv_path, 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(['map', 'success', 'path_length', 'planning_time_s'])
      for name, success, L, T in results:
        writer.writerow([name, success, f'{L:.4f}', f'{T:.4f}'])
    print(f"  saved: {csv_path}")

  # Plot and optionally save performance chart
  if labels:
    plot_performance(planner, weight, labels, lengths, times,
                     verbose=verbose, save_dir=save_dir)
