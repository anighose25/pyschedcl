#!/usr/bin/env python

from distutils.core import setup

setup(
    name='PySchedCL',
    version='1.0',
    author='Anirban Ghose, Lokesh Dokara, Srijeeta Maity',
    author_email='anighose25@gmail.com',
    packages=['pyschedcl', 'pyschedcl.test'],
    #scripts=[''],
    #url='http://',
    license='LICENSE',
    description='Useful PySchedCL-related stuff.',
    long_description=open('README.md').read(),
    platforms=[
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'matplotlib == 2.0.2',
        'Sphinx	== 1.6.5',
        'pickleshare ==	0.7.4'
    ],
)
