SRCS =	setprecision.f90 cons.f90 gammaError.f90 incgamNEG.f90
OBJS =	setprecision.o cons.o gammaError.o incgamNEG.o

ifeq ($(FC),)
    FC = gfortran
endif
ifeq ($(FFLAGS),)
    FFLAGS = -O3 -fPIC
endif
LINK_LIBS = -lquadmath -lc -lm -ldl
ifeq ($(AR),)
    AR = ar
endif

ifeq ($(OS),Windows_NT)
    BUILD = windows_build
    FFLAGS += -D WIN32
    LDFLAGS += -Wl,--out-implib,incgamNEG.lib
    LIBS = incgamNEG.dll
    SHARED_LIB = incgamNEG.dll
    STATIC_LIB = incgamNEG.dll  # We wont support static libraries in windows
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        BUILD = linux_build
        FFLAGS += -D LINUX
        LIBS = libincgamNEG.so libincgamNEG.a
        SHARED_LIB = libincgamNEG.so
        STATIC_LIB = libincgamNEG.a
    endif
    ifeq ($(UNAME_S),Darwin)
        BUILD = mac_build
        FFLAGS += -D OSX
        LIBS = libincgamNEG.so libincgamNEG.a
        SHARED_LIB = libincgamNEG.so
        STATIC_LIB = libincgamNEG.a
    endif
endif

all: shared static

shared: $(SHARED_LIB)

static: $(STATIC_LIB)

libincgamNEG.so: $(OBJS)
	$(FC) -shared $(OBJS) $(LINK_LIBS) $(FORTRAN_LDFLAGS) -o $@

libincgamNEG.a:  $(OBJS)
	$(AR) crs $@ $(OBJS) $(STATIC_LIBS)

libincgamNEG.dylib: $(OBJS)
	$(FC) -dynamiclib $(OBJS) $(LINK_LIBS) $(FORTRAN_LDFLAGS) -o $@

incgamNEG.dll:  $(OBJS)
	$(FC) -shared $(OBJS) $(LINK_LIBS) $(FORTRAN_LDFLAGS) -o $@

incgamNEG.lib:  $(OBJS)
	$(FC) -shared $(OBJS) $(LINK_LIBS) $(FORTRAN_LDFLAGS) -o $@

clean: cleantemp
	rm -f $(LIBS)

cleantemp:
	rm -f $(OBJS)  *.mod

.SUFFIXES: $(SUFFIXES) .f90

%.o:%.f90
	$(FC) $(FFLAGS) -c $<

setprecision.o: setprecision.f90
cons.o: setprecision.o cons.f90
gammaError.o: setprecision.o cons.o gammaError.f90
incgamNEG.o: setprecision.o cons.o incgamNEG.f90
