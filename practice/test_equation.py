from genepy.equation import Equation



if __name__ == "__main__":

    phenotype = "sin(X_1) + sin(sin(X_1))/3 + 4 + log(X_1*X_0)"
    #phenotype = "4.0 + X_0 + 1 + X_1"
    print(f"Starting phenotype: {phenotype}")
    equation = Equation(phenotype=phenotype) 
    print(f"Equation phenotype: {equation.phenotype}")

    import pdb;pdb.set_trace()
