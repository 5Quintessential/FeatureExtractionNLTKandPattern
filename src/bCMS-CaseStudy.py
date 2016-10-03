# CASE STUDY - BES
import FeatureX
import time

from FeatureX import *

start_time = time.time()

fx = featurex('SRS-bCMS-Specs.pdf','0-14')

fx.pre_process()
fx.extract_candidates()
fx.visualize_rel() 

print "--- %s seconds ---" % (time.time() - start_time)