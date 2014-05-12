Sambal
======

To go with the __Nutils__.

## Install

Clone `nutils/sambal` into a containing directory

```
  $ cd to/the/installation/directory
  $ mkdir sambal
  $ cd sambal
  $ git clone [https://url.to/sambal.git]
```

We can now add sambal to the PYTHONPATH so we can import from
everywhere.

```
  $ cd
```

Open `.bashrc` (linux) or `.Profile` (mac)

```
  PYTHONPATH="${PYTHONPATH}:$HOME/path/to/nutils"
  PYTHONPATH="${PYTHONPATH}:$HOME/path/to/sambal"
  export PYTHONPATH

  from nutils import *          # This does not include sambal
  from sambal import <package>  # Now we can use a sambal package
```

## Packages

  * __curve__
  * __klshell__
  * __levelset__
  * __export__ This package contains code to export in the following
    file-formats:

    - .mat files -> To load data directly into MATLAB
    - .csv files -> Plain text, comma-separated files to use with
      excel etc.

