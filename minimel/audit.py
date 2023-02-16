from vowpalwabbit import pyvw
import pathlib, collections, sys

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
    
    gold_count = collections.Counter()
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
                gold_count[ int(filt[0].split(':')[0]) ] += 1
                model.predict([ex[0]] + filt)
                i += 1
            ex = []
    
    print(dict(gold_count.most_common()), file=sys.stderr)