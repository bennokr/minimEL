from vowpalwabbit import pyvw
import pathlib

def audit(modelfile: pathlib.Path, datafile: pathlib.Path, surface:str, limit:int=1000):
    """
    Print prediction scores and model coefficients
    
    Args:
        modelfile: Model
        datafile: VW format vectorized data
    """

    model = pyvw.Workspace(
        initial_regressor=str(modelfile),
        loss_function="logistic",
        csoaa_ldf="mc",
        probabilities=True,
        testonly=True,
        audit=True,
    )
    
    ex = []
    i = 0
    for line in open(datafile):
        if i > limit:
            break
        line = line.strip()
        if line:
            ex.append(line)
        else:
            filt = [l for l in ex if f' {surface}=' in l]
            if filt:
                model.predict([ex[0]] + filt)
                i += 1
            ex = []