SRCS =	setprecision.f90 cons.f90  gammaError.f90 incgamNEG.f90
OBJS =	setprecision.o cons.o  gammaError.o incgamNEG.o

FC = gfortran
LD = gfortran
FFLAGS = -O3 -fPIC
LDFLAGS = -lquadmath -lc -lm -ldl
AR = ar

ifeq ($(OS),Windows_NT)
    BUILD = windows_build
    FFLAGS += -D WIN32
    LDFLAGS += -Wl,--out-implib,incgamNEG.lib
    LIBS = incgamNEG.dll
    SHARED_LIB = incgamNEG.dll
    STATIC_LIB = incgamNEG.dll
    ifeq ($(PROCESSOR_ARCHITEW6432),AMD64)
        FFLAGS += -D AMD64
    else
        ifeq ($(PROCESSOR_ARCHITECTURE),AMD64)
            FFLAGS += -D AMD64
        endif
        ifeq ($(PROCESSOR_ARCHITECTURE),x86)
            FFLAGS += -D IA32
        endif
    endif
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
        LIBS = libincgamNEG.dylib libincgamNEG.a
        SHARED_LIB = libincgamNEG.so
        STATIC_LIB = libincgamNEG.a
    endif
    UNAME_P := $(shell uname -p)
    ifeq ($(UNAME_P),x86_64)
        FFLAGS += -D AMD64
    endif
    ifneq ($(filter %86,$(UNAME_P)),)
        FFLAGS += -D IA32
    endif
    ifneq ($(filter arm%,$(UNAME_P)),)
        FFLAGS += -D ARM
    endif
endif

all: shared static

shared: $(SHARED_LIB)

static: $(STATIC_LIB)

libincgamNEG.so: $(OBJS)
	$(LD) -shared $(OBJS) $(LDFLAGS) -o $@

libincgamNEG.a:  $(OBJS)
	$(AR) crs $@ $(OBJS) $(STATIC_LIBS)

libincgamNEG.dylib: $(OBJS)
	$(LD) -dynamiclib $(OBJS) $(LDFLAGS) -o $@

incgamNEG.dll:  $(OBJS)
	$(LD) -shared $(OBJS) $(LDFLAGS) -o $@

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
