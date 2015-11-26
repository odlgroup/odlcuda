import os
import ycm_core
from clang_helpers import PrepareClangFlags

# Set this to the absolute path to the folder (NOT the file!) containing the
# compile_commands.json file to use that instead of 'flags'. See here for
# more details: http://clang.llvm.org/docs/JSONCompilationDatabase.html
# Most projects will NOT need to set this to anything; you can just change the
# 'flags' list of compilation flags. Notice that YCM itself uses that approach.
compilation_database_folder = os.path.dirname(os.path.abspath(__file__)) + "/build"
if compilation_database_folder:
  database = ycm_core.CompilationDatabase( compilation_database_folder )
else:
  database = None
 
 
def DirectoryOfThisScript():
  return os.path.dirname( os.path.abspath( __file__ ) )
 
 
def MakeRelativePathsInFlagsAbsolute( flags, working_directory ):
  if not working_directory:
    return list( flags )
  new_flags = []
 
  make_next_absolute = False
  remove_next = False
  remove_this = False
  path_flags = [ '-isystem', '-I', '-iquote', '--sysroot=' ]
  for flag in flags:
    new_flag = flag
 
    if remove_next:
      remove_next = False;
      remove_this = True;
 
    if make_next_absolute:
      make_next_absolute = False
      if not flag.startswith( '/' ):
        new_flag = os.path.join( working_directory, flag )
 
    for path_flag in path_flags:
      if flag == path_flag:
        make_next_absolute = True
        break
 
      if flag.startswith( path_flag ):
        path = flag[ len( path_flag ): ]
        new_flag = path_flag + os.path.join( working_directory, path )
        break
 
      # Remove the -c compiler directive (necessary to get header files compiling...)
      # YouCompleteMe seems to remove this directive correctly for the cpp files,
      # but not the header files.
      if flag == '-c':
        remove_next = True
        break
 
    if new_flag and remove_next == False and remove_this == False:
      new_flags.append( new_flag )
 
    if remove_this:
      remove_this = False
  return new_flags
 
 
def FlagsForFile( filename ):
  if database:
    # Swap .h and .cpp files 
    # Check to see if given file is a .h file, if so, swap .h and .cpp then
    # perform the lookup. This is because .h files are not included in the
    # compilation database. See: https://github.com/Valloric/YouCompleteMe/issues/174.
    # We should also resort to a minimum set of flags that work inside the
    # standard library if we can't find the compilation database entry.
    inFilename = filename
    baseFilename, fileExtension = os.path.splitext(filename)
    if (fileExtension == '.h' or fileExtension == '.t'):
        filename = baseFilename + '.cpp'
        if not os.path.isfile(filename):
            filename = os.path.dirname(inFilename)+'/dummy.cpp'
    elif (fileExtension == '.cpp'):
        pass
    else:
        filename = os.path.dirname(inFilename)+'/dummy.cpp'

    # Bear in mind that compilation_info.compiler_flags_ does NOT return a
    # python list, but a "list-like" StringVec object
    # TODO: If we don't find any flags, use the default list of flags provided.
    compilation_info = database.GetCompilationInfoForFile( filename )
    final_flags = MakeRelativePathsInFlagsAbsolute(
      compilation_info.compiler_flags_,
      compilation_info.compiler_working_dir_ )
 
  else:
    relative_to = DirectoryOfThisScript()
    final_flags = MakeRelativePathsInFlagsAbsolute( flags, relative_to )
 
  # This -x option is necessary in order to compile header files (as C++ files).
  final_flags.append('-x')
  final_flags.append('c++')
 
  # On macs, I need this in order to find the system libraries.
  # See: https://github.com/Valloric/YouCompleteMe/issues/303
  final_flags.append('-isystem')
  final_flags.append('/usr/lib/c++/v1')
 
  return {
    'flags': final_flags,
    'do_cache': True
  }
