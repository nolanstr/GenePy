from genepy.generator import Generator

generator = Generator()
generator.add_operator("add")
generator.add_operator("log")

equations = generator(20)
for equation in equations:
    print(equation)
import pdb;pdb.set_trace()

