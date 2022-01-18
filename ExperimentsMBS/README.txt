Important Note:
I renamed the .tdms files to earlier.tdms and later.tdms; specifically, timeseries.py and averagefilter.py are expecting there to be a file called later.tdms present.


Also, rulsif.py is just a library and thus has no top-level logic, so nothing will happen if you try and run it. It is used by rulsiftest.py and timeseries.py