#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision$
#EHEADER**********************************************************************


include ../config/Makefile.config

HYPRE_DIST_LS_DIRS =\
 pilut\
 ParaSails\
 Euclid

all:
	@ \
	for i in ${HYPRE_DIST_LS_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    echo "Making $$i ..."; \
	    (cd $$i && $(MAKE) $@) || exit 1; \
	  fi; \
	done

install:
	@ \
	for i in ${HYPRE_DIST_LS_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    echo "Making $$i ..."; \
	    (cd $$i && $(MAKE) $@) || exit 1; \
	  fi; \
	done

clean:
	@ \
	for i in ${HYPRE_DIST_LS_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    (cd $$i && $(MAKE) $@) || exit 1; \
	  fi; \
	done

distclean:
	@ \
	for i in ${HYPRE_DIST_LS_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    (cd $$i && $(MAKE) $@) || exit 1; \
	  fi; \
	done
