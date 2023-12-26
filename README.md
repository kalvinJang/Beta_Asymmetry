
Recommended python version for "__main__.ipynb" == 3.9
Recommended python version for "backtesting.ipynb" == 3.6

Please run the code "__main__.ipynb" to see how our team code works.
It imports our original module "pybeta.py" and uses the classes in it.

Also "__main__.ipynb" will provide you some ways to see the calculated or generated result.
Only few results will be exported to Export folder.
You can see that most of codes are under functional programming, so it has more flexibilities than procedural programming.
That is the meaning of 'only few results'.
We highly recommend you to check how our funcional codes work with diverse inputs.

After You run the "__main__.ipynb", then you should run the "bt_for_backtesting.ipynb".

Run both __main__.ipynb and bt_for_backtesting.ipynb as their precedures.
They are ready for precedure execute.
Please mail me if you have any problems during running the codes main & bt_for_backtesting.
bgkang0524@gamil.com

So if you want to see how our Beta calculation algorithmns and Generate weight matrix
please see "pybeta.py"

"__main__.ipynb" will automatically import and install all the libariest to run the "__main__.ipynb" code
it uses the following libraries
pandas, numpy, scipy, numba, numba-progress, tqdm
if the "__main__.ipynb" doesn't install the required libraries please install the libaray by
Win + R -> cmd -> pip install ooooo

"backtesting.ipynb" in Zipline Folder is not suitable for Windows. It also might not work on the Mac too.
It needs specific virtual environment to function. If you know how to do, enjoy it! (It will take some time...)
If not, not be dissapointed, reading the core code will be enough to help you to understand
our team (Delta_Mu)'s backtesting procedures. 
Also you can check the backtesting result from 'result.csv' and 'img' folder' from 'Export' folder.

But the "bt_for_backtesting.ipynb" is available for both Windows and Mac.

Lastly, We used CRSP_Data_Preprocessor.ipynb to preprocess CRSP data.
But they are not read to execute, so just looking will be fine.
CSV files with 'raw' in its name are the raw data from CRSP.
CSV files with 'PP' in its name are the preprocessed data. 

Thank you!
