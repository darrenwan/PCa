from __future__ import absolute_import

ML_TYPE = "sk-learn"
if ML_TYPE == "sk-learn":
    from .models import Model
    from .models import Sequential
    from .models import Input
else:
    # Also importable from root
    from keras import Input
    from keras import Model
    from keras import Sequential
    print("aaa")

