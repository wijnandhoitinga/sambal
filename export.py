import os
from nutils import *

class MatFile(plot.BasePlot):
  """ MatFile( name, [ ndigits ], [ index ] )

  name int      Filename without extension
  ndigits int   Number of digits for the file index
  index int     Plot number

  Note: Load in MATLAB using
    >>> load 'filename.mat'

  Example:
    with MatFile('filename') as mat:
      mat.data({'x':[0,1,2],'y':[3.1,4.5,1.2]})

  """
  def __init__( self, name, ndigits=3, index=None ):
    'constructor'

    plot.BasePlot.__init__( self, name, ndigits=ndigits, index=index )

    self.names = [ self.name + '.mat' ]
    self._data = {}

  def data( self, data ):
    'set the data'

    self._data = data

  def __exit__( self, *exc_info ):
    'exit with block'

    plot.BasePlot.__exit__( self, *exc_info )

    from scipy.io import savemat
    savemat( os.path.join( self.path, self.name), self._data, appendmat=True )

class CsvFile(plot.BasePlot):
  """ CsvFile( name, [ ndigits ], [ index ], [wrap], [ sep ], [ lbreak ] )

  name int      Filename without extension
  ndigits int   Number of digits for the file index
  index int     Plot number
  index bool    Wrap fields in quotes
  sep str       Field-separator
  lbreak str    Line-break used to start new rows

  Note: A standard comma-separated text file, you can open this using Excel,
  OpenOffice Calc or your favorite text-editor.

  Note: The order of the columns is not preserved by a standard dict
  If you want to preserve the order, store the data in collections.OrderedDict

  Note: All fields are wrapped in double-quotes, if you store a string containing
  double-quotes they are escaped by converting " into ""

  Example:
    with CsvFile('file') as csv:
      csv.data({'x':[0,1,2],'y':[3.1,4.5,1.2]})

  """
  def __init__( self, name, ndigits=3, wrap=True, index=None, sep=',', lbreak='\n' ):
    'constructor'

    plot.BasePlot.__init__( self, name, ndigits=ndigits, index=index )

    self.names = [ self.name + '.csv' ]

    self.csv = None
    self.cnt = None
    self.wrap = wrap
    self._head = ()
    self._body = []
    self._sep = sep
    self._lbreak = lbreak

  def data( self, data ):
    'set the data'

    assert type(data) == dict
    self._head = ()
    self._body = []

    self.cnt = 0
    for key in data:
      col = data[key]
      self.cnt = max(len(col), self.cnt)
      self._head += (key,)
      self._body += [list(col)]

    for i, col in enumerate(self._body):
      if len(col) < self.cnt:
        self._body[i] += ['']*(self.cnt - len(col)) # Pad the column with empty strings
        log.warning('csv > column %d padded' % i)

  def __enter__( self ):
    'enter with block'

    plot.BasePlot.__enter__( self )

    self.csv = open( os.path.join( self.path, self.name ) + '.csv', 'w' )
    return self

  def __exit__( self, *exc_info ):
    'exit with block'

    plot.BasePlot.__exit__( self, *exc_info )

    # Print table headings
    if self._head:
      if self.wrap:
        head = [ '"' + item.replace('"','""') + '"' if isinstance(item,str) else '"' + str(item) + '"' for item in self._head ]
      else:
        head = [ str(item) for item in self._head ]
      line = self._sep.join( head ) + self._lbreak
      self.csv.write( line )

    # Write the data to the csv file
    data = zip( *self._body )
    if self.cnt > 0:
      for row in data:
        if self.wrap:
          row = [ '"' + item.replace('"','""') + '"' if isinstance(item,str) else '"' + str(item) + '"' for item in row ]
        else:
          row = map(str, row)
        line = self._sep.join( row ) + self._lbreak
        self.csv.write( line )

      self.csv.close()
