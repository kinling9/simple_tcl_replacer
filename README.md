# simple tcl translator

> a simple tcl translator write in python to translate simple tcl commands in VLSI design

## requirement python module
- toml
- PLY

## usage

### one line command translate (debug mode)
```bash
python tcl_translator.py --debug ${your_config}.toml 'get_cells * -filer "ref_name=*INV*"'
```

### tcl file translate
```bash
python tcl_translator.py --process ${your_config}.toml ${your_input}.tcl ${your_output}.tcl
```

### example config file
```toml
[functions.old_func]
replace_name = "new_func"
param_count = 2 # this config have not been used yet
[functions.old_func.options]
"-old_option" = { replace_name = "-new_option" }
[functions.old_func.params]
'ori_param' = { replace_pattern = 'new_param' } # replace by re.sub
```
