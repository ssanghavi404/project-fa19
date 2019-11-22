import random

def input_generator():
    sizes = [(50, 25), (100, 50), (200, 100)]
    for numLocations, numTAs in sizes:
        f = open(str(numLocations) + '.in', 'w')
        f.write(str(numLocations) + " \n")
        f.write(str(numTAs) + " \n")
        for i in range(numLocations):
            f.write("B" + str(i) + " ")
            
        f.write('\n')
        
        for j in range(numTAs):
            building = int(random.random() * numLocations)
            f.write("B" + str(building) + " ")

        startBuilding = int(random.random() * numLocations)
        f.write("B" + str(startBuilding) + "\n")

        for _from in range(numLocations):
            for _to in range(numLocations):
                if _from == _to:
                    f.write('x ')
                else: f.write('1 ')
            f.write('\n')

        f.close()
            
if __name__ == "__main__":
    input_generator()
