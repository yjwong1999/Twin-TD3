from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

class Render(object):
    """
    plot functions to render the whole system
    """
    def __init__(self, system, canv_x = (-25, 25), canv_y = (0, 50), canv_z = (0, 60)):
        """
        docstring
        """
        plt.ion()
        self.system = system
        self.fig = plt.figure(1)
        self.pause = False
        self.t_index = 0

    def render_pause(self):
        """
        show whole system by using plt.show
        """
        plt.ion()
        ax = self.plot_config()
        # plot the position of UAV, RIS, Users & Attakers
        self.plot_entities(ax)
        self.plot_channels(ax)
        self.plot_text(ax)
        plt.show(self.fig)
        plt.cla() 
        self.pause = False
        plt.ioff() 

    def render(self, interval):
        """
        show whole system in 3D figure
        """
        plt.ion()
        ax = self.plot_config()
        # plot the position of UAV, RIS, Users & Attakers
        self.plot_entities(ax)
        self.plot_channels(ax)
        self.plot_text(ax)
        plt.pause(interval)
        plt.cla() 
        plt.ioff()

    def plot_click(self, event):
        self.pause ^= True

    def plot_config(self):
        self.fig = plt.figure(1)
        ax = plt.axes(projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim3d(-25, 25)
        ax.set_ylim3d(0, 50)
        ax.set_zlim3d(0, 60)
        ax.view_init(90, 0)
        self.fig.canvas.mpl_connect('key_press_event', self.plot_click)
        return ax    
    
    def plot_entities(self, ax):
        """
        function used in render to show the UAV, RIS, users and attackers
        """
        ax.scatter(\
        self.system.UAV.coordinate[0],\
        self.system.UAV.coordinate[1],\
        self.system.UAV.coordinate[2],\
        color='r')
        ax.text(self.system.UAV.coordinate[0],self.system.UAV.coordinate[1],self.system.UAV.coordinate[2], \
        'UAV', size=15, zorder=1, color='r') 

        ax.scatter(\
        self.system.RIS.coordinate[0],\
        self.system.RIS.coordinate[1],\
        self.system.RIS.coordinate[2],\
        color='g')
        ax.text(self.system.RIS.coordinate[0],self.system.RIS.coordinate[1],self.system.RIS.coordinate[2], \
        'RIS', size=15, zorder=1, color='g') 

        for user in self.system.user_list:
            ax.scatter(
            user.coordinate[0],\
            user.coordinate[1],\
            user.coordinate[2],\
            color='b'
            )
            text = 'user_'+str(user.index) + '\n'\
            + 'noise power(dB)    = ' + str(user.noise_power) + '\n' \
            + 'capacity          = ' + str(user.capacity) + '\n'\
            + 'secure_capacity   = ' + str(user.secure_capacity)
            
            ax.text(user.coordinate[0],user.coordinate[1],user.coordinate[2], \
            text, size=10, zorder=1, color='b') 
        for attacker in self.system.attacker_list:
            ax.scatter(
            attacker.coordinate[0],\
            attacker.coordinate[1],\
            attacker.coordinate[2],\
            color='y'
            )
            ax.text(attacker.coordinate[0],attacker.coordinate[1],attacker.coordinate[2], \
            'attacker_'+str(attacker.index) + '\n'\
            +'capacities:' + str(attacker.capacity), size=10, zorder=1, color='y') 

    def plot_channels(self, ax):
        """
        function used in render to show the H_UR, h_U_k, h_R_k
        """
        for channel in self.system.h_R_k:
            self.plot_one_channel(ax, channel, "b")
        for channel in self.system.h_R_p:
            self.plot_one_channel(ax, channel, "y")
        for channel in self.system.h_U_k:
            self.plot_one_channel(ax, channel, "b")
        for channel in self.system.h_U_p:
            self.plot_one_channel(ax, channel, "y")
            
        self.plot_one_channel(ax, self.system.H_UR, "r")
        
    def plot_one_channel(self, ax, channel, color, text = "channel"):
        """
        function used in plot channels to show only one channel
        """        
        arrow_side_coor = channel.receiver.coordinate
        point_side_coor = channel.transmitter.coordinate

        text = channel.channel_name + '\n' \
        + 'n=' + str(channel.n) \
        + '     sigma=' + str(channel.sigma) +'\n'\
        + 'PL=' + str(channel.path_loss_normal) + '\n'\
        + 'PL(dB)=' + str(channel.path_loss_dB)
        
        x = (arrow_side_coor[0] + point_side_coor[0]) / 2
        y = (arrow_side_coor[1] + point_side_coor[1]) / 2
        z = (arrow_side_coor[2] + point_side_coor[2]) / 2
        ax.text(x, y, z, text, size=10, zorder=1, color=color) 
        
        channel_arrow = Arrow3D(\
        [point_side_coor[0], arrow_side_coor[0]], \
        [point_side_coor[1], arrow_side_coor[1]], \
        [point_side_coor[2], arrow_side_coor[2]],\
        mutation_scale=20, lw = 3, arrowstyle="-|>", color=color
        )
        ax.add_artist(channel_arrow)

    def plot_text(self, ax):
        """
        used in render to polt texts
        """
        text = "pause = " + str(self.pause) + "\n"\
        + "t_index = "    + str(self.t_index)
        ax.text(0, 0, 60, text, size=10, zorder=1, color='b') 