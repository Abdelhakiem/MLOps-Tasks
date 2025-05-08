
# from mlflow.pyfunc import PythonModel, PythonModelContext

# class ModelWrapper(PythonModel):
#     def load_context(self, context: PythonModelContext):
#         import pickle
#         with open(context.artifacts["encoder"], "rb") as f:
#             self._encoder = pickle.load(f)
#         with open(context.artifacts["model"], "rb") as f:
#             self._model = pickle.load(f)

#     def predict(self, context: PythonModelContext, data):
#         preds = self._model.predict(data)
#         return [self._encoder['decoder'][val] for val in preds]

from mlflow.pyfunc import PythonModel, PythonModelContext

class ModelWrapper(PythonModel):
    def load_context(self, context: PythonModelContext):
        import pickle
        with open(context.artifacts["encoder"], "rb") as f:
            self._encoder = pickle.load(f)
        with open(context.artifacts["model"], "rb") as f:
            self._model = pickle.load(f)

    def predict(self, context: PythonModelContext, model_input):
        # model_input will be whatever was passed to predict (e.g. a DataFrame or numpy array)
        preds = self._model.predict(model_input)
        # translate encoded labels back to original
        return [self._encoder["decoder"][val] for val in preds]
