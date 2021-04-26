import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.lines as mlines


def read_file(fileName, limit=0):
    file = open(fileName, 'r')
    coords = []
    values = []
    count = 0

    angle = 0
    x, y, u, v = 0., 0., 0., 0.

    R = np.zeros((2, 2))

    for line in file.readlines():
        data = line.split()

        if count == 0:
            angle = float(data[2])
            theta = np.radians(angle)
            c, s = np.cos(theta), np.sin(theta)

            R = np.array(((c, -s), (s, c)))

            count += 1
            continue

        x, y, u, v = float(data[0]), float(
            data[1]), float(data[2]), float(data[3])

        coords.append(np.matmul(R, np.array((x, y))))
        values.append(np.matmul(R, np.array((u, v))))

        if count == limit and limit != 0:
            break

        count += 1

    return np.array(coords), np.array(values)

class WindowsVisualizer:

    def __init__(self):
        plt.ion()    
        self.fig, self.ax = plt.subplots(figsize=(30, 30))
        self.coords, self.values = read_file('src/aile')

        self.xmin = self.coords[:, 0].min()
        self.xmax = self.coords[:, 0].max()
        self.ymin = self.coords[:, 1].min()
        self.ymax = self.coords[:, 1].max()

        self.xs, self.ys, self. us, self.vs, self.speeds = self.recalibrate()

        self.grid, self.vec, self.cell = False, False, False
        self.points = []
        self.cells = self.calculate_cells()

        self.fig.canvas.mpl_connect('key_press_event', self.key_press_event)

        self.stream = None
        print("WindowsVisualizer created successfully!")

    def calculate_cells(self):
        cells = []

        tableX = np.array(self.coords[:, 0])
        tableY = np.array(self.coords[:, 1])
        tableU = np.array(self.values[:, 0])
        tableV = np.array(self.values[:, 1])

        for n in range(34):
            # Première ligne horizontale
            xs1 = tableX[n:len(tableX):35]
            ys1 = tableY[n:len(tableY):35]
            us1 = tableU[n:len(tableU):35]
            vs1 = tableU[n:len(tableV):35]

            # Deuxième ligne horizontale
            xs2 = tableX[n+1:len(tableX):35]
            ys2 = tableY[n+1:len(tableY):35]
            us2 = tableU[n+1:len(tableU):35]
            vs2 = tableU[n+1:len(tableV):35]

            for i in range(0, 87):
                x1_1, x1_2, x2_1, x2_2 = xs1[i], xs1[i+1], xs2[i], xs2[i+1]
                y1_1, y1_2, y2_1, y2_2 = ys1[i], ys1[i+1], ys2[i], ys2[i+1]
                u1_1, u1_2, u2_1, u2_2 = us1[i], us1[i+1], us2[i], us2[i+1]
                v1_1, v1_2, v2_1, v2_2 = vs1[i], vs1[i+1], vs2[i], vs2[i+1]

                cells.append((
                    (x1_1, y1_1, u1_1, v1_1),
                    (x1_2, y1_2, u1_2, v1_2),
                    (x2_2, y2_2, u2_2, v2_2),
                    (x2_1, y2_1, u2_1, v2_1)
                ))

        return np.array(cells)

    def recalibrate(self):
        x = self.coords[:, 0]
        y = self.coords[:, 1]

        u = self.values[:, 0]
        v = self.values[:, 1]

        nx, ny = len(x), len(y)
        xr = np.linspace(x.min(), x.max(), nx)
        yr = np.linspace(y.min(), y.max(), ny)

        xs, ys = np.meshgrid(xr, yr)

        px = x.flatten()
        py = y.flatten()
        pu = y.flatten()
        pv = v.flatten()

        us = griddata((px, py), pu, (xs, ys))
        vs = griddata((px, py), pv, (xs, ys))

        speeds = np.sqrt(us*us + vs*vs)

        return xs, ys, us, vs, speeds

    def profile(self, color):
        pts = self.coords[0:len(self.coords):35]

        poly = Polygon(pts)
        poly.set_color(color)
        self.ax.add_patch(poly)

    def show_meshs(self):
        tableX = self.coords[:, 0]
        tableY = self.coords[:, 1]

        for n in range(0, 35 * 87, 35):
            xs = tableX[n:n + 35]
            ys = tableY[n:n + 35]
            plt.plot(xs, ys, c=(0, 0, 0, .2))

    def show_speeds(self):
        x, y = self.coords[:, 0], self.coords[:, 1]
        u, v = self.values[:, 0], self.values[:, 1]

        self.ax.quiver(x, y, u, v, color="red", headwidth=1, scale=55)

    def show_cells(self):

        def checker(vecs):
            for i in range(len(vecs)-1):
                vec1, vec2 = vecs[i], vecs[i+1]
                if np.dot(vec1, vec2) < 0:
                    return False
            return True

        for cell in self.cells:
            x1, x2, x3, x4 = cell[:, 0]
            y1, y2, y3, y4 = cell[:, 1]
            u1, u2, u3, u4 = cell[:, 2]
            v1, v2, v3, v4 = cell[:, 3]

            vecs = [[u1, v1], [u2, v2], [u3, v3], [u4, v4]]

            if not checker(vecs):
                pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                poly = Polygon(pts)
                poly.set_color("yellow")
                self.ax.add_patch(poly)

    def calculate_lines(self):
        if(len(self.points) > 0):
            stream = self.ax.srteamplot(self.xs, self.ys, self.us, self.vs,
                                   start_points=self.points, color=self.speeds, linewidth=1, cmap="viridis")

            if self.stream is None:
                self.stream = stream
                self.colorbar = self.fig.colorbar(stream.lines)

    def show(self):
        
        plt.cla()
        self.profile("blue")
        if self.grid:
            self.show_meshs()
        if self.vec:
            self.show_speeds()
        if self.cell:
            self.show_cells()

        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)

        self.calculate_lines()

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.suptitle('Visualisation de Champs de Vecteurs')
        plt.title('Boutons: V: Speed, G: Grid, C: Cells, D: Default, S: Save')
        plt.ioff()
        plt.show()


    def key_press_event(self, event):
        print("Key press: ", event.key)
        
        plt.ion()
        if event.key == 'v':
            self.vec = not self.vec
        elif event.key == 'g':
            self.grid = not self.grid
        elif event.key == 'c':
            self.cell = not self.cell
        elif event.key == "d":
            self.grid, self.vec, self.cell = False, False, False
            
        self.show()


if __name__ == "__main__":

    windows = WindowsVisualizer()
    windows.show()

    

    
