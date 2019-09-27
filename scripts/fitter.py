import sensory_integration_time as sit
import os

if __name__=="__main__":
    options = sit.fits_module.parse_input(script_name = os.path.basename(__file__))
    sit.fits_module.main(**options)
