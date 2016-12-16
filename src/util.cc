#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sli/asarray_tstring.h>
#include <sli/tstring.h>
using namespace sli;

#include "arg_option.h"
#include "util.h"

tstring make_filename(tstring dummy, command_args *option_args)
{
  tstring my_str;
  my_str.init();
  my_str.assign(dummy.cstr());
  my_str.append("_");
  my_str.append(option_args->out_fname);
  my_str.append(".dat");

  tstring out;
  out.init();
  out.assign(option_args->out_path);
  out.append("/");
  out.append(my_str.cstr());
  return out;
}
