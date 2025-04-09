import torch
import json
from torch.utils.data import Dataset
import re
import numpy as np
import random
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt


def generateDataStrEq(
    eq, n_points=2, n_vars=3, decimals=4, supportPoints=None, min_x=0, max_x=3
):
    X = []
    Y = []
    # TODO: Need to make this faster
    for p in range(n_points):
        if supportPoints is None:
            if type(min_x) == list:
                x = []
                for _ in range(n_vars):
                    idx = np.random.randint(len(min_x))
                    x += list(
                        np.round(np.random.uniform(min_x[idx], max_x[idx], 1), decimals)
                    )
            else:
                x = list(np.round(np.random.uniform(min_x, max_x, n_vars), decimals))
            assert (
                len(x) != 0
            ), "For some reason, we didn't generate the points correctly!"
        else:
            x = supportPoints[p]

        tmpEq = eq + ""
        for nVID in range(n_vars):
            tmpEq = tmpEq.replace("x{}".format(nVID + 1), str(x[nVID]))
        y = float(np.round(eval(tmpEq), decimals))
        X.append(x)
        Y.append(y)
    return X, Y


# def processDataFiles(files):
#     text = ""
#     for f in tqdm(files):
#         with open(f, 'r') as h:
#             lines = h.read() # don't worry we won't run out of file handles
#             if lines[-1]==-1:
#                 lines = lines[:-1]
#             #text += lines #json.loads(line)
#             text = ''.join([lines,text])
#     return text


def processDataFiles(files):
    text = ""
    for f in files:
        with open(f, "r") as h:
            lines = h.read()  # don't worry we won't run out of file handles
            if lines[-1] == -1:
                lines = lines[:-1]
            # text += lines #json.loads(line)
            text = "".join([lines, text])
    return text


def tokenize_equation(eq):
    token_spec = [
        (r"\*\*"),  # exponentiation
        (r"exp"),  # exp function
        (r"[+\-*/=()]"),  # operators and parentheses
        (r"sin"),  # sin function
        (r"cos"),  # cos function
        (r"log"),  # log function
        (r"x\d+"),  # variables like x1, x23, etc.
        (r"C"),  # constants placeholder
        (r"-?\d+\.\d+"),  # decimal numbers
        (r"-?\d+"),  # integers
        (r"_"),  # padding token
    ]
    token_regex = "|".join(f"({pattern})" for pattern in token_spec)
    matches = re.finditer(token_regex, eq)
    return [match.group(0) for match in matches]


class CharDataset(Dataset):
    def __init__(
        self,
        data,
        block_size,
        tokens,
        numVars,
        numYs,
        numPoints,
        target="Skeleton",
        addVars=False,
        const_range=[-0.4, 0.4],
        xRange=[-3.0, 3.0],
        decimals=4,
        augment=False,
    ):

        data_size, vocab_size = len(data), len(tokens)
        print("data has %d examples, %d unique." % (data_size, vocab_size))

        self.stoi = {tok: i for i, tok in enumerate(tokens)}
        self.itos = {i: tok for i, tok in enumerate(tokens)}

        self.numVars = numVars
        self.numYs = numYs
        self.numPoints = numPoints

        # padding token
        self.paddingToken = "_"
        self.paddingID = self.stoi["_"]  # or another ID not already used
        self.stoi[self.paddingToken] = self.paddingID
        self.itos[self.paddingID] = self.paddingToken

        self.threshold = [-1000, 1000]

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data  # it should be a list of examples
        self.target = target
        self.addVars = addVars

        self.const_range = const_range
        self.xRange = xRange
        self.decimals = decimals
        self.augment = augment

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        # grab an example from the data
        chunk = self.data[idx]  # sequence of tokens including x, y, eq, etc.

        try:
            chunk = json.loads(chunk)  # convert the sequence tokens to a dictionary
        except Exception as e:
            print("Couldn't convert to json: {} \n error is: {}".format(chunk, e))
            # try the previous example
            idx = idx - 1
            idx = idx if idx >= 0 else 0
            chunk = self.data[idx]
            chunk = json.loads(chunk)  # convert the sequence tokens to a dictionary

        # find the number of variables in the equation
        printInfoCondition = random.random() < 0.0000001
        eq = chunk[self.target]
        if printInfoCondition:
            print(f"\nEquation: {eq}")
        vars = re.finditer("x[\d]+", eq)
        numVars = 0
        for v in vars:
            v = v.group(0).strip("x")
            v = eval(v)
            v = int(v)
            if v > numVars:
                numVars = v

        if self.target == "Skeleton" and self.augment:
            threshold = 5000
            # randomly generate the constants
            cleanEqn = ""
            for chr in eq:
                if chr == "C":
                    # genereate a new random number
                    chr = "{}".format(
                        np.random.uniform(self.const_range[0], self.const_range[1])
                    )
                cleanEqn += chr

            # update the points
            nPoints = np.random.randint(
                *self.numPoints
            )  # if supportPoints is None else len(supportPoints)
            try:
                if printInfoCondition:
                    print("Org:", chunk["X"], chunk["Y"])

                X, y = generateDataStrEq(
                    cleanEqn,
                    n_points=nPoints,
                    n_vars=self.numVars,
                    decimals=self.decimals,
                    min_x=self.xRange[0],
                    max_x=self.xRange[1],
                )

                # replace out of threshold with maximum numbers
                y = [e if abs(e) < threshold else np.sign(e) * threshold for e in y]

                # check if there is nan/inf/very large numbers in the y
                conditions = (
                    (np.isnan(y).any() or np.isinf(y).any())
                    or len(y) == 0
                    or (abs(min(y)) > threshold or abs(max(y)) > threshold)
                )
                if not conditions:
                    chunk["X"], chunk["Y"] = X, y

                if printInfoCondition:
                    print("Evd:", chunk["X"], chunk["Y"])
            except Exception as e:
                # for different reason this might happend including but not limited to division by zero
                print(
                    "".join(
                        [
                            f"We just used the original equation and support points because of {e}. ",
                            f"The equation is {eq}, and we update the equation to {cleanEqn}",
                        ]
                    )
                )

        # encode every character in the equation to an integer
        # < is SOS, > is EOS
        if self.addVars:
            dix = [self.stoi[s] for s in "<" + str(numVars) + ":" + eq + ">"]
        else:
            eq_tokens = tokenize_equation(eq)
            if self.addVars:
                token_seq = ["<", str(numVars), ":", *eq_tokens, ">"]
            else:
                token_seq = ["<", *eq_tokens, ">"]
            dix = [self.stoi[tok] for tok in token_seq]

        inputs = dix[:-1]
        outputs = dix[1:]

        # add the padding to the equations
        paddingSize = max(self.block_size - len(inputs), 0)
        paddingList = [self.paddingID] * paddingSize
        inputs += paddingList
        outputs += paddingList

        # make sure it is not more than what should be
        inputs = inputs[: self.block_size]
        outputs = outputs[: self.block_size]

        points = torch.zeros(self.numVars + self.numYs, self.numPoints - 1)
        for idx, xy in enumerate(zip(chunk["X"], chunk["Y"])):

            if not isinstance(xy[0], list) or not isinstance(
                xy[1], (list, float, np.float64)
            ):
                print(f"Unexpected types: {type(xy[0])}, {type(xy[1])}")
                continue  # Skip if types are incorrect

            # don't let to exceed the maximum number of points
            if idx >= self.numPoints - 1:
                break

            x = xy[0]
            x = x + [0] * (max(self.numVars - len(x), 0))  # padding

            y = [xy[1]] if type(xy[1]) == float or type(xy[1]) == np.float64 else xy[1]

            y = y + [0] * (max(self.numYs - len(y), 0))  # padding
            p = x + y  # because it is only one point
            p = torch.tensor(p)
            # replace nan and inf
            p = torch.nan_to_num(
                p,
                nan=self.threshold[1],
                posinf=self.threshold[1],
                neginf=self.threshold[0],
            )

            points[:, idx] = p

        points = torch.nan_to_num(
            points,
            nan=self.threshold[1],
            posinf=self.threshold[1],
            neginf=self.threshold[0],
        )

        inputs = torch.tensor(inputs, dtype=torch.long)
        outputs = torch.tensor(outputs, dtype=torch.long)
        numVars = torch.tensor(numVars, dtype=torch.long)
        return inputs, outputs, points, numVars


# Relative Mean Square Error
def relativeErr(y, yHat, info=False, eps=1e-5):
    yHat = np.reshape(yHat, [1, -1])[0]
    y = np.reshape(y, [1, -1])[0]
    if len(y) > 0 and len(y) == len(yHat):
        err = ((yHat - y)) ** 2 / np.linalg.norm(y + eps)
        if info:
            for _ in range(5):
                i = np.random.randint(len(y))
                # print("yPR,yTrue:{},{}, Err:{}".format(yHat[i], y[i], err[i]))
    else:
        err = 100

    return np.mean(err)


def lossFunc(constants, eq, X, Y, eps=1e-5):
    err = 0
    eq = eq.replace("C", "{}").format(*constants)

    for x, y in zip(X, Y):
        eqTemp = eq + ""
        if type(x) == np.float32:
            x = [x]
        for i, e in enumerate(x):
            # make sure e is not a tensor
            if type(e) == torch.Tensor:
                e = e.item()
            eqTemp = eqTemp.replace("x{}".format(i + 1), str(e))
        try:
            yHat = eval(eqTemp)
        except:
            # print("Exception has been occured! EQ: {}, OR: {}".format(eqTemp, eq))
            yHat = 100
        try:
            # handle overflow
            err += relativeErr(y, yHat)  # (y-yHat)**2
        except:
            # print(
            #    "Exception has been occured! EQ: {}, OR: {}, y:{}-yHat:{}".format(
            #        eqTemp, eq, y, yHat
            #    )
            # )
            err += 10

    err /= len(Y)
    return err


def get_predicted_skeleton(generated_tokens, train_dataset: CharDataset):
    predicted_tokens = generated_tokens.cpu().numpy()
    predicted = "".join([train_dataset.itos[int(idx)] for idx in predicted_tokens])
    predicted = predicted.strip(train_dataset.paddingToken).split(">")
    predicted = predicted[0] if len(predicted[0]) >= 1 else predicted[1]
    predicted = predicted.strip("<").strip(">")
    predicted = predicted.replace("Ce", "C*e")

    return predicted


def sample_skeleton(model, points, variables, train_dataset, batch_size, ddim_step=20):
    """Sample skeletons from the model using DDIM or DDPM."""
    return model.sample(
        points, variables, train_dataset, batch_size=batch_size, ddim_step=ddim_step
    )


def fit_constants(predicted_skeleton, X, Y):
    """Fit constants in the predicted skeleton using optimization."""
    c = [1.0 for i, x in enumerate(predicted_skeleton) if x == "C"]
    b = [(-2, 2) for _, x in enumerate(predicted_skeleton) if x == "C"]
    predicted = predicted_skeleton
    if len(c) != 0:
        try:
            cHat = minimize(lossFunc, c, args=(predicted_skeleton, X, Y), bounds=b)
            if cHat.success and cHat.fun != float("inf"):
                predicted = predicted_skeleton.replace("C", "{}").format(*cHat.x)
            else:
                raise ValueError(
                    f"Invalid predicted equation or optimization failed: {predicted_skeleton}"
                )
        except Exception as e:
            raise ValueError(
                f"Error fitting constants: {e}, Equation: {predicted_skeleton}"
            )
    return predicted


def evaluate_equation(eq, xs, target=True):
    """Evaluate an equation at given points xs."""
    SAFE_GLOBALS = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
        "sqrt": math.sqrt,
        "abs": abs,
        "pow": pow,
        "__builtins__": {},
    }
    try:
        eq_tmp = eq.replace(" ", "").replace("\n", "")
        for i, x in enumerate(xs):
            eq_tmp = eq_tmp.replace(f"x{i+1}", str(x))
            if "," in eq_tmp:
                raise ValueError("There is a , in the equation!")
        y = eval(eq_tmp, SAFE_GLOBALS)
        y = 0 if np.isnan(y) else y
        y = 100 if np.isinf(y) else y
    except Exception as e:
        if target:
            # print(f"TA: Evaluation failed for equation: {eq_tmp}, Reason: {e}")
            pass
        else:
            # print(f"PR: Evaluation failed for equation: {eq_tmp}, Reason: {e}")
            pass
        y = 100
    return y


def evaluate_sample(target_eq, predicted_eq, test_points):
    """Evaluate target and predicted equations over test points."""
    Ys, Yhats = [], []
    for xs in test_points:
        Ys.append(evaluate_equation(target_eq, xs, target=True))
        Yhats.append(evaluate_equation(predicted_eq, xs, target=False))
    return Ys, Yhats


def sample_and_evaluate(
    model, points, variables, train_dataset, target_eq, t, ddim_step=None
):
    """Sample a skeleton, fit constants, evaluate, and compute error."""
    predicted_skeleton = sample_skeleton(
        model, points, variables, train_dataset, batch_size=1, ddim_step=ddim_step
    )[0]

    predicted = fit_constants(predicted_skeleton, t["X"], t["Y"])

    Ys, Yhats = evaluate_sample(target_eq, predicted, t["XT"])

    err = relativeErr(Ys, Yhats, info=True)
    return predicted_skeleton, predicted, err


def sample_and_select_best(
    model,
    points,
    variables,
    train_dataset,
    target_eq,
    t,
    num_tests,
    ddim_step=None,
):
    """Sample num_tests times and select the best sample based on error."""
    best_err = 10000000
    best_predicted_skeleton = "C"
    best_predicted = "C"

    for _ in range(num_tests):
        predicted_skeleton, predicted, err = sample_and_evaluate(
            model, points, variables, train_dataset, target_eq, t, ddim_step
        )
        if err < best_err:
            best_err = err
            best_predicted_skeleton = predicted_skeleton
            best_predicted = predicted

    return best_predicted_skeleton, best_predicted, best_err


def plot_and_save_results(
    resultDict, fName, pconf, titleTemplate, textTest, modelKey="SymbolicDiffusion"
):
    """Plot cumulative error distribution and save results to a file.
    Args:
        resultDict: dict with results (e.g., {'err': [], 'trg': [], 'prd': []})
        fName: str, output file name
        pconf: PointNetConfig object with numberofVars
        titleTemplate: str, template for plot title
        textTest: list of test data
        modelKey: str, key for the model in resultDict
    """
    if isinstance(resultDict, dict):
        num_eqns = len(resultDict[fName][modelKey]["err"])
        num_vars = pconf.numberofVars
        title = titleTemplate.format(num_eqns, num_vars)

        models = list(
            key
            for key in resultDict[fName].keys()
            if len(resultDict[fName][key]["err"]) == num_eqns
        )
        lists_of_error_scores = [
            resultDict[fName][key]["err"]
            for key in models
            if len(resultDict[fName][key]["err"]) == num_eqns
        ]
        linestyles = ["-", "dashdot", "dotted", "--"]

        eps = 0.00001
        y, x, _ = plt.hist(
            [
                np.log([max(min(x + eps, 1e5), 1e-5) for x in e])
                for e in lists_of_error_scores
            ],
            label=models,
            cumulative=True,
            histtype="step",
            bins=2000,
            density=True,
            log=False,
        )
        y = np.expand_dims(y, 0)

        plt.figure(figsize=(15, 10))
        for idx, m in enumerate(models):
            plt.plot(x[:-1], y[idx] * 100, linestyle=linestyles[idx], label=m)

        plt.legend(loc="upper left")
        plt.title(title)
        plt.xlabel("Log of Relative Mean Square Error")
        plt.ylabel("Normalized Cumulative Frequency")

        name = "{}.png".format(fName.split(".txt")[0])
        plt.savefig(name)
        plt.close()

        with open(fName, "w", encoding="utf-8") as o:
            for i in range(num_eqns):
                err = resultDict[fName][modelKey]["err"][i]
                eq = resultDict[fName][modelKey]["trg"][i]
                predicted = resultDict[fName][modelKey]["prd"][i]
                print(f"Test Case {i}.")
                print(f"Target: {eq}\nSkeleton: {predicted}")
                print(f"Err: {err}\n")

                o.write(f"Test Case {i}/{len(textTest)-1}.\n")
                o.write(f"{eq}\n")
                o.write(f"{modelKey}:\n")
                o.write(f"{predicted}\n{err}\n\n")

            avg_err = np.mean(resultDict[fName][modelKey]["err"])
            o.write(f"Avg Err: {avg_err}\n")
            print(f"Avg Err: {avg_err}")
