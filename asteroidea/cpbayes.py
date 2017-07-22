import subprocess


def prepare_input():
    """Save the input file in disk"""
    return False


def search():
    args = "bin/search mld_scores.txt".split()
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()
    output = output.decode('utf-8')
    return output


def parse_output(results):
    start = res.rfind('backtracking solution, time')
    end = len(res)
    lines = results[start:end].split('\n')
    for line in lines:
        if 'ordering' in line:
            head_var = find_between(line, '=', '<')
            body_vars = find_between(line, '{', '}').split()


def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
