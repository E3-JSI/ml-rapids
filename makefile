config = release
# config = debug
build = build/$(config)

ifeq ($(config), debug)
	debug = -g
else
	# debug =
endif

src = code/src
test = code/test
targetfile = streamdm
flags = -std=c++11 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE64 -O3 -DUNIX $(debug) 

# Core library
sourcefiles_lib = \
    $(wildcard $(src)/core/*.cpp) \
    $(wildcard $(src)/utils/*.cpp) \
    $(wildcard $(src)/streams/*.cpp) \
    $(wildcard $(src)/evaluation/*.cpp) \
    $(wildcard $(src)/tasks/*.cpp) \
    $(wildcard $(src)/learners/*.cpp) \
    $(wildcard $(src)/learners/Classifiers/Bayes/*.cpp) \
    $(wildcard $(src)/learners/Classifiers/Bayes/observer/*.cpp) \
    $(wildcard $(src)/learners/Classifiers/Functions/*.cpp) \
    $(wildcard $(src)/learners/Classifiers/Meta/*.cpp) \
    $(wildcard $(src)/learners/Classifiers/Trees/*.cpp)

includepath_lib = -I$(src)
flags_lib = -JSON_DLL_BUILD -DSTREAMDM_EXPORTS -fPIC -shared $(flags)

# Python library
sourcefiles_py = $(src)/$(targetfile)_wrap.cxx $(sourcefiles_lib)
includepath_py = \
	-I$(src) \
	-I$(PYTHON_INCLUDE) \
	-I$(NUMPY_INCLUDE)
flags_py = -JSON_DLL_BUILD -DSTREAMDM_EXPORTS -fPIC -shared $(flags)

# Test
sourcefiles_test = $(wildcard $(test)/*.cpp)
includepath_test = -I$(src)
flags_test = $(flags)

all: py

py:
	mkdir -p $(build)
	swig -c++ -python -o $(src)/$(targetfile)_wrap.cxx -outdir $(build) $(src)/$(targetfile).i
	g++ $(sourcefiles_py) $(includepath_py) $(flags_py) -o $(build)/_$(targetfile).so
	cp $(test)/test.py $(build)

lib:
	mkdir -p $(build)
	g++ $(sourcefiles_lib) $(includepath_lib) $(flags_lib) -o $(build)/$(targetfile).so
	g++ $(sourcefiles_test) $(includepath_test) -L$(build) -l:$(targetfile).so $(flags_test) -o $(build)/test

clean:
	rm -rf $(build)
	rm -f $(src)/$(targetfile)_wrap.cxx
