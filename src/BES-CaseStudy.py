# CASE STUDY - BES
import FeatureX
import time

from FeatureX import *

start_time = time.time()

fx = featurex('SRS-BECS-2007.pdf','0-20')

fx.pre_process()
fx.extract_candidates()
fx.visualize_rel() 

print "--- %s seconds ---" % (time.time() - start_time)