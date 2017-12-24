import numpy as np
import lib.logger.logger as logger
DEBUG_MODE = False


class IniParser(object):

    def __init__(self):
        self.log = logger.get_logger('IniParser')

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)

    def add_section(self, txt, name, config=None):
        if len(txt) > 0:
            section_name = txt + '.' + name
        else:
            section_name = name

        if config is not None:
            config.add_section(section_name)

        return section_name

    def set_to_config(self, do_save_none, section, config, key, val):
        if val is not None:
            config.set(section, key, str(val))
        else:
            if do_save_none:
                config.set(section, key, 'None')

    def parse_from_config(self, obj_instance, override_mode, section, parser, key, val_type):
        try:
            val_as_str = parser.get(section, key)
            val = val_type(val_as_str)
            if DEBUG_MODE:
                self.log.info(
                    'Parse: section: {} , key: {}, val_str: {}, val_type: {}'.format(section, key, val_as_str, val))

            if isinstance(val, bool):
                val = parser.getboolean(section, key)
            elif isinstance(val, np.ndarray):
                val = eval(val_as_str)
            else:
                if val_as_str == 'None':
                    val = None
            if val is not None:
                setattr(obj_instance, key, val)
        except:
            # key do not exist: error in load mode and O.K. in override mode
            if not override_mode:
                # Error in load mode
                raise Exception('Non existing key: [{}].{} '.format(section, key))
            else:
                # In override mode, do not set any new value
                if DEBUG_MODE:
                    self.log.warn(
                        'Parse: section: {} , key: {}. KEY NOT FOUND'.format(section, key))

                return

class FrozenClass(IniParser):
    __isfrozen = False

    def __init__(self):
        self.log = logger.get_logger('FrozenIniParser')
        super(FrozenClass,self).__init__()

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            err_str = 'Can not set value to a frozen class: (key={},val={})'.format(key, value)
            self.log.error(err_str)
            raise TypeError(err_str)
        else:
            super(FrozenClass, self).__setattr__(key, value)

    def _freeze(self):
        self.__isfrozen = True
