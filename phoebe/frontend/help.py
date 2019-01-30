
from pydoc import pager
from phoebe import conf

class HelpEntry(object):
    def __init__(self, msg, include_choices=False, **kwargs):
        self.meta = kwargs
        self.msg = msg
        self.include_choices = include_choices

    def is_match(self, **kwargs):
        for k,v in kwargs.items():
            if k=='entry':
                # then we need to deal with these in order of preference
                if 'choice' in self.meta.keys():
                    k = 'choice'
                else:
                    k = 'qualifier'
            if self.meta.get(k, v) != v:
                # here we essentially ignore anything in the passed kwargs
                # that has not been defined by the HelpEntry
                return False
        return True

_help = []
_help += [HelpEntry(qualifier='period', kind='orbit', msg='test message for parameter:period@binary')]
_help += [HelpEntry(qualifier='period', kind='star', msg='test message for parameter:period@star')]

_help += [HelpEntry(qualifier='atm', msg='test message for parameter:atm', include_choices=True)]
_help += [HelpEntry(choice='ck2004', qualifier='atm', msg='test message for choice:ck2004')]
_help += [HelpEntry(choice='blackbody', qualifier='atm', msg='test message for choice:blackbody')]


def phoebehelp(entry, empty_if_none=False, **kwargs):
    """
    search available help messages and return matching entry.
    """

    if isinstance(entry, str):
        kwargs['entry'] = entry

    elif 'Parameter' in entry.__class__.__name__:
        # here we avoid importing Parameter and using isinstance because
        # parameters.py needs to import help and we can't have a race condition
        for k,v in entry.meta.items():
            if k=='qualifier':
                k='entry'
            kwargs.setdefault(k, v)
    else:
        # then access the docstring by calling python's builtin help
        help(entry)
        return

    matches = [h for h in _help if h.is_match(**kwargs)]
    if len(matches) == 0:
        if empty_if_none:
            return ''
        else:
            raise ValueError("no help matches found")
    elif len(matches) > 1:
        raise ValueError("more than 1 help match found: {}".format([h.meta for h in matches]))
    else:
        match = matches[0]

        if match.include_choices:
            kwargs['qualifier'] = kwargs.pop('entry')
            choices = [h for h in _help if h.is_match(level='choice', **kwargs)]

            msg = match.msg
            msg += "\nCHOICES:"
            for choice in choices:
                if choice == match:
                    # our filter above doesn't exclude the acutal parameter,
                    # so let's avoid listing it again here
                    continue
                msg += "\n\t{}".format(choice.msg)

        else:
            msg = match.msg

        if conf._in_interactive_session:
            return pager(msg)
        else:
            return msg
