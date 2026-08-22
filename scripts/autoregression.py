"""A small autoregression model for time-series forecasting, using NumPy only.

WHY THIS EXISTS

The measurements in this project are a time series: one value after another, in
order, where each value is strongly related to the value just before it. A plain
linear regression on the time step ignores that relationship. It draws one
straight line through the whole run and asks only "how far along are we?".

Autoregression asks a different and more natural question: "given the last few
measurements, what comes next?". The model is

    y[t] = c + a1*y[t-1] + a2*y[t-2] + ... + ap*y[t-p]

where p is the number of past values used, called the order or the number of
lags. Fitting it is ordinary least squares on columns of shifted values, so it
needs nothing beyond NumPy. The statsmodels package would produce the same
coefficients; we avoid the dependency because this project runs on a small
machine.

TWO WAYS IT IS USED HERE

    One step ahead   Predict the very next measurement from the measured past.
                     This is what the error scores in the evaluation report.

    Many steps ahead Feed each prediction back in as history and keep going, to
                     answer "when will this cross the danger line?". Errors
                     compound the further out it runs, which is stated plainly
                     wherever the forecast is printed.

RUN BOUNDARIES

A dataset here holds several complete runs one after another. The last value of
one run tells you nothing about the first value of the next, so the training
data is built per run and never spans a boundary.
"""

import numpy as np


class AutoRegression:
    """Predict the next value in a series from the previous `lags` values."""

    def __init__(self, lags=3):
        if lags < 1:
            raise ValueError("lags must be at least 1")
        self.lags = lags
        self.coefficients = None      # a1 ... ap
        self.intercept = 0.0

    # --------------------------------------------------------------- training
    def _design(self, series):
        """Turn one series into (rows of past values, the value that followed)."""
        series = np.asarray(series, dtype=float)
        if len(series) <= self.lags:
            return np.empty((0, self.lags)), np.empty(0)
        rows = [series[t - self.lags:t][::-1] for t in range(self.lags, len(series))]
        return np.array(rows), series[self.lags:]

    def fit(self, series_list):
        """Fit on a list of series, one per run, so no row crosses a run boundary."""
        if isinstance(series_list, (list, tuple)) and len(series_list) \
                and np.ndim(series_list[0]) == 0:
            series_list = [series_list]      # a single bare series was passed

        blocks = [self._design(series) for series in series_list]
        blocks = [(X, y) for X, y in blocks if len(y)]
        if not blocks:
            raise ValueError(
                f"not enough data to fit an AR({self.lags}) model: every run is "
                f"{self.lags} rows or shorter")

        X = np.vstack([block[0] for block in blocks])
        y = np.concatenate([block[1] for block in blocks])

        # A column of ones gives the model its intercept.
        design = np.hstack([X, np.ones((len(X), 1))])
        solution, *_ = np.linalg.lstsq(design, y, rcond=None)
        self.coefficients = solution[:-1]
        self.intercept = float(solution[-1])
        return self

    # ------------------------------------------------------------- prediction
    def _check_fitted(self):
        if self.coefficients is None:
            raise ValueError("call fit() before predicting")

    def predict_next(self, history):
        """Predict the single next value from the last `lags` observations."""
        self._check_fitted()
        recent = np.asarray(history, dtype=float)[-self.lags:][::-1]
        if len(recent) < self.lags:
            raise ValueError(f"need {self.lags} past values, got {len(recent)}")
        return float(np.dot(self.coefficients, recent) + self.intercept)

    def predict_one_step(self, series):
        """One-step-ahead predictions for a whole series.

        Returns (actual, predicted) for every position from `lags` onward. Each
        prediction uses only measured values that came before it, never a value
        from the future and never its own earlier prediction.
        """
        self._check_fitted()
        series = np.asarray(series, dtype=float)
        actual, predicted = [], []
        for t in range(self.lags, len(series)):
            predicted.append(self.predict_next(series[:t]))
            actual.append(series[t])
        return np.array(actual), np.array(predicted)

    def forecast(self, history, steps):
        """Predict several steps ahead, feeding each prediction back as history."""
        self._check_fitted()
        window = list(np.asarray(history, dtype=float)[-self.lags:])
        out = []
        for _ in range(int(steps)):
            nxt = self.predict_next(window)
            out.append(nxt)
            window = window[1:] + [nxt]
        return np.array(out)

    def steps_until(self, history, threshold, limit=500):
        """How many steps ahead the series is forecast to cross `threshold`.

        Returns None when the forecast does not reach the threshold within
        `limit` steps, which is the honest answer for a series that has levelled
        off or is falling.
        """
        self._check_fitted()
        window = list(np.asarray(history, dtype=float)[-self.lags:])
        for step in range(1, int(limit) + 1):
            nxt = self.predict_next(window)
            if nxt >= threshold:
                return step
            window = window[1:] + [nxt]
        return None

    def describe(self):
        """The fitted equation, written out for a reader."""
        self._check_fitted()
        terms = " + ".join(f"{c:.3f}*y[t-{i + 1}]"
                           for i, c in enumerate(self.coefficients))
        return f"y[t] = {self.intercept:.2f} + {terms}"
