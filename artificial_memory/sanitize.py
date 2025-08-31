import json

def sanitize_response(response: str) -> dict:
    lines = response.replace('\n', ',').split(',')
    print(lines)

    sanitized_lines: list[str] = []
    for line in lines:
        if '//' in line:
            sanitized_lines.append(line.split('//')[0].strip())
        elif '#' in line:
            sanitized_lines.append(line.split('#')[0].strip())
        else:
            sanitized_lines.append(line.strip())

    filtered_lines: list[tuple[int, str]] = [(index, line.split('"time":')[1].strip()) for index, line in enumerate(sanitized_lines) if line.lstrip().startswith('"time":')]

    for index, nb in filtered_lines:
        try:
            if any(not char.isdigit() and not char in '+-*/' for char in nb):
                raise ValueError("No numbers found in the string")
            result = eval(nb)
            sanitized_lines[index] = sanitized_lines[index] = f'"time":{result},'
        except Exception as e:
            raise Exception(f"Could not evaluate: {nb}, error: {e}")

    dic = json.loads(''.join(sanitized_lines))
    return dic

if __name__ == "__main__":
    with open('output.txt', 'r') as file:
        response = file.read()
    sanitized_response = sanitize_response(response)
    print(sanitized_response)
