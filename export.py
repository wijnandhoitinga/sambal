from nutils import *

class MatFile:
  """ MatFile( name )

  Note: Load in MATLAB using
    >>> load 'file.mat'

  Example:
    with MatFile( 'file.mat') as mat:
      mat.data({'x':[0,1,2],'y':[3.1,4.5,1.2]})

  """
  def __init__( self, name ):
    self.name = name
    self._data = {}

  def data( self, data ):
    self._data = data

  def __enter__( self ):
    return self

  def __exit__( self, type, value, traceback ):
    from scipy.io import savemat
    savemat( self.name, self._data )

class CsvFile:
  """ CsvFile( name, [ sep ], [ lbreak ] )

  sep str     Field-separator
  lbreak str  Line-break used to start new rows

  Note: A standard comma-separated text file, you can open this using Excel,
  OpenOffice Calc or your favorite text-editor.

  Note: The order of the columns is not preserved by a standard dict
  If you want to preserve the order, store the data in collections.OrderedDict

  Note: All fields are wrapped in double-quotes, if you store a string containing
  double-quotes they are escaped by converting " into ""

  Example:
    with CsvFile( 'file.csv') as csv:
      csv.data({'x':[0,1,2],'y':[3.1,4.5,1.2]})

  """
  def __init__( self, name, sep=',', lbreak='\n' ):
    self.name = name
    self.csv = None
    self.cnt = None
    self._head = ()
    self._body = []
    self._sep = sep
    self._lbreak = lbreak

  def data( self, data ):
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
        self._body[i] += ['']*(self.cnt - len(col)) # Padd the col with empty strings
        log.warning('csv > column %d padded' % i)

  def __enter__( self ):
    self.csv = open( self.name, 'w' )
    return self

  def __exit__( self, type, value, traceback ):
    # Print head
    if self._head:
      head = [ '"' + item.replace('"','""') + '"' if isinstance(item,str) else '"' + str(item) + '"' for item in self._head ]
      line = self._sep.join( head ) + self._lbreak
      self.csv.write( line )


    # Write the data to the csv file
    data = zip( *self._body )
    if self.cnt > 0:
      for row in data:
        row = [ '"' + item.replace('"','""') + '"' if isinstance(item,str) else '"' + str(item) + '"' for item in row ]
        line = self._sep.join( row ) + self._lbreak
        self.csv.write( line )

      self.csv.close()
