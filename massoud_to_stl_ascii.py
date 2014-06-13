import numpy as np

def parse_ascii_fepoint(f, outfile):
    """expects a filelike object, and returns a nx12 array. One row for every facet in the STL file."""
    scalar = 1
    stack = []
    IDs = []
    facets = []
    connectivity = []

    line = f.readline() # Skip header lines
    line = f.readline()
    line = f.readline()

    i = line.strip().split(',')[-5]
    i_strt = i.find('=')

    i = int(i[i_strt+1:])

    line = f.readline() # Read first line

    line_num = 1

    while line:

        if line_num <= i: # Store vertices
            stack.extend(map(float, line.strip().split()[0:3]))
            IDs.extend(map(int, line.strip().split()[3:4]))
            facets.append(stack)
            stack = []
            line = f.readline()
            line_num+=1

        else: # Store Connectivity data
            stack.extend(map(int, line.strip().split()[0:3]))
            connectivity.append(stack)
            stack = []
            line = f.readline()

    with open(outfile, 'w') as ofile:

        ofile.write('solid ascii_io_exported_from_Pointwise \n')
        for line in connectivity:
            ofile.write('  facet normal 0.0 0.0 0.0 \n')
            ofile.write('    outerloop \n')
            for ID in line:
                vertices = facets[ID-1]
                ofile.write('      vertex %.16f %.16f %.16f %d \n' % (vertices[0]*scalar, vertices[1]*scalar, vertices[2]*scalar, IDs[ID-1]))
            ofile.write('    endloop \n')
            ofile.write('  endfacet \n')

        ofile.write('endsolid ascii_io_exported_from_Pointwise \n')

    return np.array(facets)

outer_cowl_file = open('noz01_massoud_body1.dat', 'rb')
inner_cowl_file = open('noz01_massoud_body2.dat', 'rb')
outer_shroud_file = open('noz01_massoud_body3.dat', 'rb')
inner_shroud_file = open('noz01_massoud_body4.dat', 'rb')
centerbody_file = open('noz01_massoud_body5.dat','rb')

facets = parse_ascii_fepoint(outer_cowl_file, 'OuterCowl_ASCII.stl')
facets = parse_ascii_fepoint(inner_cowl_file, 'InnerCowl_ASCII.stl')
facets = parse_ascii_fepoint(outer_shroud_file, 'OuterShroud_ASCII.stl')
facets = parse_ascii_fepoint(inner_shroud_file, 'InnerShroud_ASCII.stl')
facets = parse_ascii_fepoint(centerbody_file, 'Centerbody_ASCII.stl')