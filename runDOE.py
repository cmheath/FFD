''' Top level assembly for the Lean Direct Injection (LDI) design space analysis '''

# --- Inherent python/system level imports
import os

# --- OpenMDAO main and library imports
from openmdao.main.api import Assembly, SequentialWorkflow, set_as_top
from openmdao.lib.drivers.api import DOEdriver
from openmdao.lib.doegenerators.api import OptLatinHypercube, Uniform, FullFactorial
from openmdao.lib.casehandlers.api import DBCaseIterator, DBCaseRecorder, case_db_to_dict
import numpy as np
import pylab as p

# --- OpenMDAO component imports
from geometrycomponent import GeometryComp

DOE_OUT_DB = 'DOE_Output.db'

class Analysis(Assembly):

    def __init__(self):
        super(Analysis, self).__init__()
        
        # --------------------------------------------------------------------------- #
        # --- Instantiate LHC DOE Driver
        # --------------------------------------------------------------------------- #  
        self.add('doe_driver', DOEdriver())  
        self.doe_driver.DOEgenerator = Uniform(num_samples = 5)
        self.doe_driver.workflow = SequentialWorkflow()
       
        # --------------------------------------------------------------------------- #
        # --- Instantiate Geometry Component
        # --------------------------------------------------------------------------- #  
        self.add('geometry', GeometryComp()) 

        # 1--- Top Level Workflow
        self.driver.workflow.add(['doe_driver']) 
        self.doe_driver.workflow.add(['geometry'])
    
        # --------------------------------------------------------------------------- #
        # --- Add parameters to DOE driver 
        # --------------------------------------------------------------------------- #         
        self.doe_driver.add_parameter('geometry.dC1P_X')
        self.doe_driver.add_parameter('geometry.dC2P_X')
        self.doe_driver.add_parameter('geometry.dC3P_X')
        self.doe_driver.add_parameter('geometry.dC4P_X')

        self.doe_driver.add_parameter('geometry.dC1P_R')
        self.doe_driver.add_parameter('geometry.dC2P_R')
        self.doe_driver.add_parameter('geometry.dC3P_R')
        self.doe_driver.add_parameter('geometry.dC4P_R')

        self.doe_driver.add_parameter('geometry.dC1S_X')
        self.doe_driver.add_parameter('geometry.dC2S_X')
        self.doe_driver.add_parameter('geometry.dC3S_X')
        self.doe_driver.add_parameter('geometry.dC4S_X')

        self.doe_driver.add_parameter('geometry.dC1S_R')
        self.doe_driver.add_parameter('geometry.dC2S_R')
        self.doe_driver.add_parameter('geometry.dC3S_R')
        self.doe_driver.add_parameter('geometry.dC4S_R')
       
        self.doe_driver.add_parameter('geometry.dC1S_T')
        self.doe_driver.add_parameter('geometry.dC2S_T')
        self.doe_driver.add_parameter('geometry.dC3S_T')

        self.doe_driver.add_parameter('geometry.dC1C_X')
        self.doe_driver.add_parameter('geometry.dC2C_X')
        self.doe_driver.add_parameter('geometry.dC3C_X')
        self.doe_driver.add_parameter('geometry.dC4C_X')

        self.doe_driver.add_parameter('geometry.dC1C_R')
        self.doe_driver.add_parameter('geometry.dC2C_R')
        self.doe_driver.add_parameter('geometry.dC3C_R')
        self.doe_driver.add_parameter('geometry.dC4C_R')
       
        self.doe_driver.add_parameter('geometry.dC1C_T')
        self.doe_driver.add_parameter('geometry.dC2C_T')
        self.doe_driver.add_parameter('geometry.dC3C_T')        
       
if __name__ == '__main__':
    
        
    top_level_analysis = set_as_top(Analysis())  
    top_level_analysis.run()    

    X = []
    Y = []


    print 'opening file'

    lines = [line.strip() for line in open('designs.txt')]

    # print 'lines', lines
    count = 0

    for line in lines:
        data = []
        for i in line.split():
            data.append(float(i))
        
        if count % 2 == 0:
            X = np.array(data)
        else:
            Y = np.array(data)
            
            if count < 100*10:
                p.scatter(X,Y, c = np.random.rand(3,1), linewidth = 0.5)
            else:
                p.plot(X,Y,c = np.random.rand(3,1), linewidth = 0.5)                

        count+=1


    p.show()


    # --------------------------------END---------------------------------------- #        
        

    

       