# -*- coding: utf-8 -*-
"""Class_Imblance.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oTJaI7DGALy4q7VwSAFMOYLuU4fGQA-D
"""

groups = wine.groupby('quality').size()
groups.plot.bar()