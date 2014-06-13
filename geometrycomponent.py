''' OpenMDAO geometry component '''

# --- Inherent python/system level imports
import os
import sys
import shutil
import time

# --- External python library imports (i.e. matplotlib, numpy, scipy)
from tempfile import TemporaryFile
import math
from numpy import linspace, array, zeros, ones, vstack, cos, sin, pi, outer
from matplotlib import pylab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pylab as p

# --- OpenMDAO imports
from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Float

# --- Local imports
import stl as stl
from ffd_axisymetric import Body, Shell
from stl_group import STLGroup

class GeometryComp(Component):
    ''' OpenMDAO component for Geometry Handling '''
   
    # -----------------------------------------------------
    # --- Initialize Input Design Parameters and Ranges ---
    # -----------------------------------------------------
    dC0P_X = Float(0.0, low = 0.0, high = 0.0, iotype ='in', desc ='x position control point 1')
    dC1P_X = Float(1.0, low = -1.5, high = 1.0, iotype ='in', desc ='x position control point 2')
    dC2P_X = Float(-1.0, low = -1.5, high = 1.0, iotype ='in', desc ='x position control point 3')
    dC3P_X = Float(0.0, low = -2.0, high = 1.5, iotype ='in', desc ='x position control point 4')
    dC4P_X = Float(0.0, low = -4.0, high = 0.5, iotype ='in', desc ='x position control point 5')
    
    dC0P_R = Float(0.0, low = 0.0, high = 0.0, iotype ='in', desc ='r position control point 1')
    dC1P_R = Float(0.0, low = -1.0, high = 1.0, iotype ='in', desc ='r position control point 2')
    dC2P_R = Float(0.0, low = -1.0, high = 1.0, iotype ='in', desc ='r position control point 3')
    dC3P_R = Float(0.0, low = -1.0, high = 1.0, iotype ='in', desc ='r position control point 4')
    dC4P_R = Float(0.0, low = 0.0, high = 0.0, iotype ='in', desc ='r position control point 5')    
    
    dC0S_X = Float(0.0, low = 0.0, high = 0.0, iotype ='in', desc ='x position shroud centerline control point 1')
    dC1S_X = Float(0.0, low = -1.0, high = 1.0, iotype ='in', desc ='x position shroud centerline control point 2')
    dC2S_X = Float(0.0, low = -1.0, high = 1.0, iotype ='in', desc ='x position shroud centerline control point 3')
    dC3S_X = Float(0.0, low = -1.0, high = 1.0, iotype ='in', desc ='x position shroud centerline control point 4')
    dC4S_X = Float(0.0, low = -2.0, high = 0.5, iotype ='in', desc ='x position shroud centerline control point 5')    
    
    dC0S_R = Float(0.0, low = 0.0, high = 0.0, iotype ='in', desc ='r position shroud centerline control point 1')
    dC1S_R = Float(0.0, low = -1.0, high = 1.0, iotype ='in', desc ='r position shroud centerline control point 2')
    dC2S_R = Float(0.0, low = -1.0, high = 1.0, iotype ='in', desc ='r position shroud centerline control point 3')
    dC3S_R = Float(0.0, low = -1.0, high = 1.0, iotype ='in', desc ='r position shroud centerline control point 4')
    dC4S_R = Float(0.0, low = -1.0, high = 1.0, iotype ='in', desc ='r position shroud centerline control point 5')    
    
    dC0S_T = Float(0.0, low = 0.0, high = 0.0, iotype ='in', desc ='r position shroud thickness control point 1')
    dC1S_T = Float(0.0, low = -0.25, high = 1.5, iotype ='in', desc ='r position shroud thickness control point 2')
    dC2S_T = Float(0.0, low = -0.75, high = 1.5, iotype ='in', desc ='r position shroud thickness control point 3')
    dC3S_T = Float(0.0, low = 0.0, high = 1.5, iotype ='in', desc ='r position shroud thickness control point 4')
    dC4S_T = Float(0.0, low = 0.0, high = 0.0, iotype ='in', desc ='r position shroud thickness control point 5') 

    dC0C_X = Float(0.0, low = 0.0, high = 0.0, iotype ='in', desc ='x position cowl centerline control point 1')
    dC1C_X = Float(0.0, low = -1.0, high = 1.0, iotype ='in', desc ='x position cowl centerline control point 2')
    dC2C_X = Float(0.0, low = -1.0, high = 1.0, iotype ='in', desc ='x position cowl centerline control point 3')
    dC3C_X = Float(0.0, low = -1.0, high = 1.0, iotype ='in', desc ='x position cowl centerline control point 4')
    dC4C_X = Float(0.0, low = -2.0, high = 0.5, iotype ='in', desc ='x position cowl centerline control point 5')    
    
    dC0C_R = Float(0.0, low = 0.0, high = 0.0, iotype ='in', desc ='r position cowl centerline control point 1')
    dC1C_R = Float(0.0, low = -1.5, high = 1.5, iotype ='in', desc ='r position cowl centerline control point 2')
    dC2C_R = Float(0.0, low = -1.5, high = 1.5, iotype ='in', desc ='r position cowl centerline control point 3')
    dC3C_R = Float(0.0, low = -1.5, high = 1.5, iotype ='in', desc ='r position cowl centerline control point 4')
    dC4C_R = Float(0.0, low = -1.0, high = 1.0, iotype ='in', desc ='r position cowl centerline control point 5')    
    
    dC0C_T = Float(0.0, low = 0.0, high = 0.0, iotype ='in', desc ='r position cowl thickness control point 1')
    dC1C_T = Float(0.0, low = -1.0, high = 1.5, iotype ='in', desc ='r position cowl thickness control point 2')
    dC2C_T = Float(0.0, low = -0.5, high = 1.5, iotype ='in', desc ='r position cowl thickness control point 3')
    dC3C_T = Float(0.0, low = 0.0, high = 1.5, iotype ='in', desc ='r position cowl thickness control point 4')
    dC4C_T = Float(0.0, low = 0.0, high = 0.0, iotype ='in', desc ='r position cowl thickness control point 5') 
    
    geom = STLGroup()
    
    def __init__(self, *args, **kwargs):

        # ----------------------------------------------
        # --- Constructor for the Geometry Component ---
        # ----------------------------------------------
        super(GeometryComp, self).__init__(*args, **kwargs) 

        start_time = time.time()

        this_dir, this_filename = os.path.split(__file__)
        centerbody_file = os.path.join(this_dir, 'Centerbody_ASCII.stl')      
        centerbody = stl.STL(centerbody_file)

        outer_cowl_file = os.path.join(this_dir, 'OuterCowl_ASCII.stl')        
        outer_cowl = stl.STL(outer_cowl_file)

        inner_cowl_file = os.path.join(this_dir, 'InnerCowl_ASCII.stl')        
        inner_cowl = stl.STL(inner_cowl_file)

        outer_shroud_file = os.path.join(this_dir, 'OuterShroud_ASCII.stl')
        outer_shroud = stl.STL(outer_shroud_file)

        inner_shroud_file = os.path.join(this_dir, 'InnerShroud_ASCII.stl')
        inner_shroud = stl.STL(inner_shroud_file)

        print "STL Load Time: ", time.time()-start_time
        start_time = time.time()        

        n_c = 5

        C_x = np.array([0.0, 1.27, 2.54, 3.81, 8.09743]) 
        C_r = np.zeros((len(C_x),))
        control_points = np.array(zip(C_x,C_r))

        plug = Body(centerbody, controls=control_points, x_ref=0.15, r_ref=0.025)

        C_x = np.array([0.726, 300.0, 1.524, 2.54, 4.16007])
        C_r = np.zeros((len(C_x),))
        control_points = np.array(zip(C_x,C_r))

        cowl = Shell(outer_cowl, inner_cowl, center_line_controls=control_points, thickness_controls=control_points, x_ref=0.15, r_ref=0.02)

        C_x = np.array([0.0, 0.762, 1.524, 2.54, 3.80468])
        C_r = np.zeros((len(C_x),))
        control_points = np.array(zip(C_x,C_r))        
        
        shroud = Shell(outer_shroud, inner_shroud, center_line_controls=control_points, thickness_controls=control_points, x_ref=0.15, r_ref=0.02)

        print "Bspline Compute Time: ", time.time()-start_time
        start_time = time.time()

        self.geom.add(plug, name="plug")
        self.geom.add(cowl, name="cowl")
        self.geom.add(shroud, name="shroud")

        print "Geometry Object Building: ", time.time()-start_time
        start_time = time.time()
        
    def execute(self):
    
        n_c = 5
        
        # --- Move centerbody control points
        dCP_X = array([self.dC0P_X, self.dC1P_X, self.dC2P_X, self.dC3P_X, self.dC4P_X])
        dCP_R = array([self.dC0P_R, self.dC1P_R, self.dC2P_R, self.dC3P_R, self.dC4P_R])
        dCP_C = array(zip(dCP_X, dCP_R))

        # --- Move shroud centerline control points
        dCS_X = array([self.dC0S_X, self.dC1S_X, self.dC2S_X, self.dC3S_X, self.dC4S_X])
        dCS_R = array([self.dC0S_R, self.dC1S_R, self.dC2S_R, self.dC3S_R, self.dC4S_R])
        dCS_C = array(zip(dCS_X, dCS_R))

        # --- Move shroud thickness control points (only in r direction)
        dCS_X = np.zeros((n_c,))
        dCS_R = array([self.dC0S_T, self.dC1S_T, self.dC2S_T, self.dC3S_T, self.dC4S_T])
        dCS_T = array(zip(dCS_X, dCS_R))

        # --- Move cowl centerline control points
        dCC_X = array([self.dC0C_X, self.dC1C_X, self.dC2C_X, self.dC3C_X, self.dC4C_X])
        dCC_R = array([self.dC0C_R, self.dC1C_R, self.dC2C_R, self.dC3C_R, self.dC4C_R])
        dCC_C = array(zip(dCC_X, dCC_R))

        # --- Move cowl thickness control points (only in r direction)
        dCC_X = np.zeros((n_c,))
        dCC_R = array([self.dC0C_T, self.dC1C_T, self.dC2C_T, self.dC3C_T, self.dC4C_T])
        dCC_T = array(zip(dCC_X, dCC_R))

        # baseline_profile = self.geom.project_profile()

        start_time = time.time()

        # self.geom.deform(plug=dCP_C, shroud=(dCS_C, dCS_T),cowl=(dCC_C, dCC_T))

        # print "Run Time: ", time.time()-start_time
        # start_time = time.time()

        # profile = self.geom.project_profile()        
            
        self.geom.writeSTL('deformed_geom.stl', ascii=True)

        print "STL Write Time: ", time.time()-start_time
        start_time = time.time()

        self.geom.writeFEPOINT('model.tec.1.sd1')

        print "FEPOINT Write Time: ", time.time()-start_time
        start_time = time.time()

        # for point_set in profile:
        #     X = point_set[:,0]
        #     Y = point_set[:,1]

        #     print 'max X = ', np.max(X)

        #     with open('designs.txt', 'a') as f:

        #         f.writelines(["%s " % item  for item in X])
        #         f.writelines("\n")
        #         f.writelines(["%s " % item  for item in Y])
        #         f.writelines("\n")

        
if __name__ == "__main__":
    import logging
    from openmdao.main.api import enable_console

    enable_console()
    logging.getLogger().setLevel(logging.DEBUG)
    
    # -------------------------
    # --- Default Test Case ---
    # ------------------------- 
    Geom_Comp = GeometryComp()
    
    Geom_Comp.run()



