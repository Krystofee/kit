import os
import pickle
from copy import copy
from difflib import SequenceMatcher
from enum import Enum
from typing import Tuple, List, Iterable, Optional
from uuid import uuid4

import click

INIT_COMMIT_PARENT = 'INIT'


def not_empty(file_path):
    return os.path.exists(file_path) and os.stat(file_path).st_size > 0


class OperationType(Enum):
    INSERT = 'INSERT'
    REPLACE = 'REPLACE'
    DELETE = 'DELETE'

    @classmethod
    def get_type(cls, opcode):
        STR_TO_OP = {
            'insert': cls.INSERT,
            'replace': cls.REPLACE,
            'delete': cls.DELETE,
        }
        return STR_TO_OP.get(opcode)


class Operation:
    op_type: OperationType
    at: Tuple[int, int]
    value: str

    def __init__(self, op_type, at, value):
        self.op_type = op_type
        self.at = at
        self.value = value

    def __str__(self):
        return f'<Operation {self.op_type.value:7} ({self.index_offset()}) at {self.at} value "{self.value}">'

    def __repr__(self):
        return str(self)

    def index_offset(self):
        if self.op_type == OperationType.INSERT:
            return len(self.value)
        elif self.op_type == OperationType.REPLACE:
            return len(self.value) - (self.at[1] - self.at[0])
        elif self.op_type == OperationType.DELETE:
            return (self.at[1] - self.at[0]) * -1
        return 0

    def get_at(self, offset):
        return self.at[0] + offset, self.at[1] + offset

    def apply(self, to: str, index_offset=0):
        offset_at = self.get_at(index_offset)

        if self.op_type == OperationType.INSERT:
            return to[:offset_at[0]] + self.value + to[offset_at[1]:]
        elif self.op_type == OperationType.REPLACE:
            return to[:offset_at[0]] + self.value + to[offset_at[1]:]
        elif self.op_type == OperationType.DELETE:
            return to[:offset_at[0]] + to[offset_at[1]:]


class ChangeType(Enum):
    ADD = 'ADD'
    EDIT = 'EDIT'
    MOVE = 'MOVE'
    REMOVE = 'REMOVE'


class Change:
    file_path: str
    type: ChangeType
    operations: Optional[Iterable[Operation]]

    def __init__(self, file_path, type, operations=None):
        self.file_path = file_path
        self.type = type
        self.operations = operations

    def __repr__(self):
        return f'<Change {self.file_path} {self.type} {", ".join(map(str, self.operations))}>'

    def apply(self, file):
        result = copy(file)
        index_offset = 0
        for op in self.operations:
            result = op.apply(result, index_offset=index_offset)
            index_offset += op.index_offset()
        return result


class Commit:
    id: str
    message: str
    parents: List[str]
    changes: List[Change]

    def __init__(self, id: str, message: str, parents: Optional[List[str]], changes: List[Change]):
        self.id = id
        self.message = message
        self.parents = parents or []
        self.changes = changes

    def __repr__(self):
        return f'<Commit {self.id[:6]} {self.message} from {", ".join([x[:6] for x in self.parents]) or "-"} {len(self.changes)}>'

    @property
    def short_id(self):
        return self.id[:6]

    def to_string(self, verbose=True):
        result = f"Commit: {self.message}\n"
        result += f"{self.id} parents {', '.join([x[:6] for x in self.parents]) or '-'}\n"

        if not len(self.changes):
            result += '- empty commit\n'

        for change in self.changes:
            result += f'- {change.type.value:6} {change.file_path}\n'

        return result

    def apply(self, file_dict):
        for change in self.changes:
            if change.file_path in file_dict:
                file_dict[change.file_path] = change.apply(file_dict.get(change.file_path))

        return file_dict


class CommitStorage:
    def __init__(self, repo):
        self.repo = repo
        self.commits = {}

    def load(self):
        if not_empty(self.repo.commit_storage_path):
            with open(self.repo.commit_storage_path, 'rb') as file:
                self.commits = pickle.load(file)

    def save(self):
        with open(self.repo.commit_storage_path, 'wb') as file:
            pickle.dump(self.commits, file)

    def store(self, commit):
        self.commits[commit.id] = commit

    def get(self, id):
        return self.commits.get(id)


class WorkIndex:
    tracked: set
    staged: set

    def __init__(self, repo):
        self.repo = repo
        self.tracked = set()
        self.staged = set()

    def load(self):
        if not_empty(self.repo.working_index_path):
            with open(self.repo.working_index_path, 'rb') as file:
                self.tracked, self.staged = pickle.load(file)

    def save(self):
        print(f'... saved working index')
        if not os.path.exists(self.repo.working_index_path):
            with open(self.repo.working_index_path, 'xb'):
                pass

        with open(self.repo.working_index_path, 'wb') as file:
            data = (self.tracked, self.staged)
            pickle.dump(data, file)

    def add(self, file_path):
        print(f'... added {file_path} to tracked files')
        self.tracked.add(file_path)

    def stage(self, file_path):
        print(f'... added {file_path} to staged files')
        self.staged.add(file_path)

    def clear(self):
        print(f'... cleared working index')
        self.staged = set()
        self.tracked = set()
        self.save()


class Repository:
    root: str

    def __init__(self, root='.kit'):
        self.root = root
        self.storage = CommitStorage(self)
        self.work_index = WorkIndex(self)

    def load(self):
        self.storage.load()
        self.work_index.load()

    @property
    def head_path(self):
        return os.path.join(self.root, 'HEAD')

    @property
    def commit_storage_path(self):
        return os.path.join(self.root, 'commits')

    @property
    def working_index_path(self):
        return os.path.join(self.root, 'working_index')

    @property
    def head_ref(self):
        return self._get_head_ref()

    def exists(self):
        return os.path.exists(self.root) and os.path.exists(self.head_path)

    def init(self):
        os.mkdir(self.root)

        # Init head file
        with open(self.head_path, 'x') as head_file:
            head_file.write('')

        # Commit file
        with open(self.commit_storage_path, 'xb') as file:
            file.write(b'')

        # Working index file
        with open(self.working_index_path, 'xb') as file:
            file.write(b'')

        self.create_commit(Commit(INIT_COMMIT_PARENT, 'Initial commit', [], []))

    def _get_head_ref(self):
        with open(self.head_path, 'r') as head:
            head = head.readlines()
            if head:
                return head[0]

    def _update_head_ref(self, commit_ref):
        with open(self.head_path, 'w') as head:
            head.write(commit_ref)

        print(f'... moved head to {commit_ref}')

    def create_commit(self, commit):
        print(f'... storing commit {commit.short_id}')

        self.storage.store(commit)
        self.storage.save()
        self._update_head_ref(commit.id)
        self.work_index.clear()


class CommitIndex:
    def __init__(self, storage, head_id):
        self.storage = storage
        self.head_id = head_id
        self.indexed_files = None

    def build(self):
        head_commit = self.storage.get(self.head_id)
        print('... building index from', head_commit)
        self.indexed_files = self._build_recursive(head_commit)

    def _build_recursive(self, commit):
        if not commit:
            return {}

        print('... in', commit)

        result = {}
        for parent in commit.parents:
            parent_commit = self.storage.get(parent)
            result.update(self._build_recursive(parent_commit))

        for change in commit.changes:
            if change.type == ChangeType.ADD:
                result[change.file_path] = commit.id
            if change.type == ChangeType.REMOVE:
                del result[change.file_path]

        return result

    def get_commit_path(self, commit_id, grand_parent_commit_id):
        commit = self.storage.get(commit_id)
        grand_parent_commit = self.storage.get(grand_parent_commit_id)

        if not commit or not grand_parent_commit:
            return []

        if commit_id == grand_parent_commit_id:
            return [commit]

        return self._get_commit_path_recursive(commit, grand_parent_commit)

    def _get_commit_path_recursive(self, commit, grand_parent_commit):
        for parent_id in commit.parents:
            parent = self.storage.get(parent_id)

            if parent_id == grand_parent_commit.id:
                return [commit, [parent]]

            parent_result = self._get_commit_path_recursive(parent, grand_parent_commit)
            if parent_result:
                return [commit, parent_result]

        return None

    def restore_file(self, file_path):
        commit_path = self.get_commit_path(self.head_id, self.indexed_files[file_path])

        if not commit_path:
            return None

        result = self._restore_file_recursive(file_path, '', commit_path)
        return result[file_path]

    @staticmethod
    def _restore_file_recursive(file_path, file_content, commit_path):
        if len(commit_path) == 1:
            return commit_path[0].apply({file_path: file_content})

        commit = commit_path[0]
        new_commit_path = commit_path[1]

        return commit.apply(CommitIndex._restore_file_recursive(file_path, file_content, new_commit_path))


def init_fresh_repo(repo):
    repo.init()

    c1 = Commit(str(uuid4()), 'c1', None, [
        Change('a.txt', ChangeType.ADD, [
            Operation(OperationType.INSERT, (0, 0), 'Hello world!')],
               ),
        Change('b.txt', ChangeType.ADD, [
            Operation(OperationType.INSERT, (0, 0), 'Second file')],
               ),
    ])
    c2 = Commit(str(uuid4()), 'c2', [c1.id], [
        Change('a.txt', ChangeType.EDIT, [
            Operation(OperationType.REPLACE, (6, 12), 'Krystofee!'),
            Operation(OperationType.INSERT, (12, 12), ' How are you?'),
        ]),
    ])
    c3 = Commit(str(uuid4()), 'c3', [c2.id], [
        Change('b.txt', ChangeType.REMOVE, [
            Operation(OperationType.INSERT, (0, 0), 'Hello world!')],
               ),
        Change('test/x.txt', ChangeType.ADD, [
            Operation(OperationType.INSERT, (0, 0), 'test nested file')],
               ),
    ])
    c4 = Commit(str(uuid4()), 'c4', [c3.id], [
        Change('a.txt', ChangeType.EDIT, [
            Operation(OperationType.INSERT, (0, 0), '... ')],
               ),
        Change('b.txt', ChangeType.ADD, [
            Operation(OperationType.INSERT, (0, 0), 'NEW B!!!')],
               ),
    ])

    storage = CommitStorage(repo)
    storage.store(c1)
    storage.store(c2)
    storage.store(c3)
    storage.store(c4)
    storage.save()

    repo._update_head_ref(c4.id)

def print_indexed_files(commit_index):
    print('Indexed files:')
    print(commit_index.indexed_files)
    print()

    for indexed_file in commit_index.indexed_files.keys():
        print('restored', indexed_file, 'is', commit_index.restore_file(indexed_file))


def get_operations(original, new):
    matcher = SequenceMatcher(lambda x: x == " ", original, new)

    operations = []
    for opcode in matcher.get_opcodes():
        operation_type = OperationType.get_type(opcode[0])
        if operation_type:
            operations.append(
                Operation(
                    OperationType.get_type(opcode[0]), (opcode[1], opcode[2]), new[opcode[3]:opcode[4]]
                )
            )
    return operations


def create_commit(storage, work_index, against_commit, message):
    if against_commit == None:
        pass # Test

    changes = []

    # tracked_files = set(commit_index.indexed_files.keys())
    staged_files = work_index.staged

    visited_files = set()

    if against_commit:
        commit_index = CommitIndex(storage, against_commit)
        commit_index.build()

        for file_path in commit_index.indexed_files.keys():
            visited_files.add(file_path)
            if os.path.exists(file_path):
                with open(file_path) as file:
                    original_content = commit_index.restore_file(file_path)
                    current_content = ''.join(file.readlines())
                    operations = get_operations(original_content, current_content)
                    if operations:
                        changes.append(Change(file_path, ChangeType.EDIT, operations))
            else:
                changes.append(Change(file_path, ChangeType.REMOVE, []))

    for file_path in staged_files - set(visited_files):
        with open(file_path) as file:
            current_content = ''.join(file.readlines())
            changes.append(Change(file_path, ChangeType.ADD, [Operation(OperationType.INSERT, (0, 0), current_content)]))

    parents = []
    if against_commit:
        parents.append(against_commit)
    else:
        parents.append(INIT_COMMIT_PARENT)

    return Commit(str(uuid4()), message, parents, changes)

# repo = Repository()
# init_fresh_repo(repo)
# repo.load()
#
# commit_index = CommitIndex(repo.storage, repo.head_ref)
# commit_index.build()
#
# work_index = WorkIndex.load(repo)
#
# changes = create_commit(repo.storage, work_index, repo.head_ref)
# commit = Commit(str(uuid4()), [repo.head_ref], changes)
#
# print(commit)
# print(commit.changes)

# repo.create_commit(commit)


@click.group()
def kit_cli():
    pass


@click.command()
def init():
    print('Initializing repository in .')
    repo = Repository()
    if repo.exists():
        print('... repository already exists')
    else:
        repo.init()
        print('... initialized repository at .')


@click.command()
@click.argument('file_path', required=True)
def add(file_path):
    repo = Repository()
    repo.load()
    repo.work_index.stage(file_path)
    repo.work_index.save()


@click.command()
@click.option('--message', required=True, help="Attach commit message")
def commit(message):
    repo = Repository()
    repo.load()

    commit = create_commit(repo.storage, repo.work_index, repo.head_ref, message)

    print('...created commit')
    print()
    print(commit.to_string())

    repo.create_commit(commit)


@click.command()
def log():
    repo = Repository()
    repo.load()

    commit_index = CommitIndex(repo.storage, repo.head_ref)
    history = commit_index.get_commit_path(repo.head_ref, INIT_COMMIT_PARENT)

    if not len(history):
        print('Nothing to display.')

    while history:
        commit = history[0]
        print(commit.to_string())

        if len(history) <= 1:
            break

        history = history[1]


@click.command()
def status():
    repo = Repository()
    repo.load()

    if repo.work_index.staged:
        print('Staged files:')
        for file in repo.work_index.staged:
            print('- ', file)
    else:import os
import pickle
from copy import copy
from difflib import SequenceMatcher
from enum import Enum
from typing import Tuple, List, Iterable, Optional
from uuid import uuid4

import click

INIT_COMMIT_PARENT = 'INIT'


def not_empty(file_path):
    return os.path.exists(file_path) and os.stat(file_path).st_size > 0


class OperationType(Enum):
    INSERT = 'INSERT'
    REPLACE = 'REPLACE'
    DELETE = 'DELETE'

    @classmethod
    def get_type(cls, opcode):
        STR_TO_OP = {
            'insert': cls.INSERT,
            'replace': cls.REPLACE,
            'delete': cls.DELETE,
        }
        return STR_TO_OP.get(opcode)


class Operation:
    op_type: OperationType
    at: Tuple[int, int]
    value: str

    def __init__(self, op_type, at, value):
        self.op_type = op_type
        self.at = at
        self.value = value

    def __str__(self):
        return f'<Operation {self.op_type.value:7} ({self.index_offset()}) at {self.at} value "{self.value}">'

    def __repr__(self):
        return str(self)

    def index_offset(self):
        if self.op_type == OperationType.INSERT:
            return len(self.value)
        elif self.op_type == OperationType.REPLACE:
            return len(self.value) - (self.at[1] - self.at[0])
        elif self.op_type == OperationType.DELETE:
            return (self.at[1] - self.at[0]) * -1
        return 0

    def get_at(self, offset):
        return self.at[0] + offset, self.at[1] + offset

    def apply(self, to: str, index_offset=0):
        offset_at = self.get_at(index_offset)

        if self.op_type == OperationType.INSERT:
            return to[:offset_at[0]] + self.value + to[offset_at[1]:]
        elif self.op_type == OperationType.REPLACE:
            return to[:offset_at[0]] + self.value + to[offset_at[1]:]
        elif self.op_type == OperationType.DELETE:
            return to[:offset_at[0]] + to[offset_at[1]:]


class ChangeType(Enum):
    ADD = 'ADD'
    EDIT = 'EDIT'
    MOVE = 'MOVE'
    REMOVE = 'REMOVE'


class Change:
    file_path: str
    type: ChangeType
    operations: Optional[Iterable[Operation]]

    def __init__(self, file_path, type, operations=None):
        self.file_path = file_path
        self.type = type
        self.operations = operations

    def __repr__(self):
        return f'<Change {self.file_path} {self.type} {", ".join(map(str, self.operations))}>'

    def apply(self, file):
        result = copy(file)
        index_offset = 0
        for op in self.operations:
            result = op.apply(result, index_offset=index_offset)
            index_offset += op.index_offset()
        return result


class Commit:
    id: str
    message: str
    parents: List[str]
    changes: List[Change]

    def __init__(self, id: str, message: str, parents: Optional[List[str]], changes: List[Change]):
        self.id = id
        self.message = message
        self.parents = parents or []
        self.changes = changes

    def __repr__(self):
        return f'<Commit {self.id[:6]} {self.message} from {", ".join([x[:6] for x in self.parents]) or "-"} {len(self.changes)}>'

    @property
    def short_id(self):
        return self.id[:6]

    def to_string(self, verbose=True):
        result = f"Commit: {self.message}\n"
        result += f"{self.id} parents {', '.join([x[:6] for x in self.parents]) or '-'}\n"

        if not len(self.changes):
            result += '- empty commit\n'

        for change in self.changes:
            result += f'- {change.type.value:6} {change.file_path}\n'

        return result

    def apply(self, file_dict):
        for change in self.changes:
            if change.file_path in file_dict:
                file_dict[change.file_path] = change.apply(file_dict.get(change.file_path))

        return file_dict


class CommitStorage:
    def __init__(self, repo):
        self.repo = repo
        self.commits = {}

    def load(self):
        if not_empty(self.repo.commit_storage_path):
            with open(self.repo.commit_storage_path, 'rb') as file:
                self.commits = pickle.load(file)

    def save(self):
        with open(self.repo.commit_storage_path, 'wb') as file:
            pickle.dump(self.commits, file)

    def store(self, commit):
        self.commits[commit.id] = commit

    def get(self, id):
        return self.commits.get(id)


class WorkIndex:
    tracked: set
    staged: set

    def __init__(self, repo):
        self.repo = repo
        self.tracked = set()
        self.staged = set()

    def load(self):
        if not_empty(self.repo.working_index_path):
            with open(self.repo.working_index_path, 'rb') as file:
                self.tracked, self.staged = pickle.load(file)

    def save(self):
        print(f'... saved working index')
        if not os.path.exists(self.repo.working_index_path):
            with open(self.repo.working_index_path, 'xb'):
                pass

        with open(self.repo.working_index_path, 'wb') as file:
            data = (self.tracked, self.staged)
            pickle.dump(data, file)

    def add(self, file_path):
        print(f'... added {file_path} to tracked files')
        self.tracked.add(file_path)

    def stage(self, file_path):
        print(f'... added {file_path} to staged files')
        self.staged.add(file_path)

    def clear(self):
        print(f'... cleared working index')
        self.staged = set()
        self.tracked = set()
        self.save()


class Repository:
    root: str

    def __init__(self, root='.kit'):
        self.root = root
        self.storage = CommitStorage(self)
        self.work_index = WorkIndex(self)

    def load(self):
        self.storage.load()
        self.work_index.load()

    @property
    def head_path(self):
        return os.path.join(self.root, 'HEAD')

    @property
    def commit_storage_path(self):
        return os.path.join(self.root, 'commits')

    @property
    def working_index_path(self):
        return os.path.join(self.root, 'working_index')

    @property
    def head_ref(self):
        return self._get_head_ref()

    def exists(self):
        return os.path.exists(self.root) and os.path.exists(self.head_path)

    def init(self):
        os.mkdir(self.root)

        # Init head file
        with open(self.head_path, 'x') as head_file:
            head_file.write('')

        # Commit file
        with open(self.commit_storage_path, 'xb') as file:
            file.write(b'')

        # Working index file
        with open(self.working_index_path, 'xb') as file:
            file.write(b'')

        self.create_commit(Commit(INIT_COMMIT_PARENT, 'Initial commit', [], []))

    def _get_head_ref(self):
        with open(self.head_path, 'r') as head:
            head = head.readlines()
            if head:
                return head[0]

    def _update_head_ref(self, commit_ref):
        with open(self.head_path, 'w') as head:
            head.write(commit_ref)

        print(f'... moved head to {commit_ref}')

    def create_commit(self, commit):
        print(f'... storing commit {commit.short_id}')

        self.storage.store(commit)
        self.storage.save()
        self._update_head_ref(commit.id)
        self.work_index.clear()


class CommitIndex:
    def __init__(self, storage, head_id):
        self.storage = storage
        self.head_id = head_id
        self.indexed_files = None

    def build(self):
        head_commit = self.storage.get(self.head_id)
        print('... building index from', head_commit)
        self.indexed_files = self._build_recursive(head_commit)

    def _build_recursive(self, commit):
        if not commit:
            return {}

        print('... in', commit)

        result = {}
        for parent in commit.parents:
            parent_commit = self.storage.get(parent)
            result.update(self._build_recursive(parent_commit))

        for change in commit.changes:
            if change.type == ChangeType.ADD:
                result[change.file_path] = commit.id
            if change.type == ChangeType.REMOVE:
                del result[change.file_path]

        return result

    def get_commit_path(self, commit_id, grand_parent_commit_id):
        commit = self.storage.get(commit_id)
        grand_parent_commit = self.storage.get(grand_parent_commit_id)

        if not commit or not grand_parent_commit:
            return []

        if commit_id == grand_parent_commit_id:
            return [commit]

        return self._get_commit_path_recursive(commit, grand_parent_commit)

    def _get_commit_path_recursive(self, commit, grand_parent_commit):
        for parent_id in commit.parents:
            parent = self.storage.get(parent_id)

            if parent_id == grand_parent_commit.id:
                return [commit, [parent]]

            parent_result = self._get_commit_path_recursive(parent, grand_parent_commit)
            if parent_result:
                return [commit, parent_result]

        return None

    def restore_file(self, file_path):
        commit_path = self.get_commit_path(self.head_id, self.indexed_files[file_path])

        if not commit_path:
            return None

        result = self._restore_file_recursive(file_path, '', commit_path)
        return result[file_path]

    @staticmethod
    def _restore_file_recursive(file_path, file_content, commit_path):
        if len(commit_path) == 1:
            return commit_path[0].apply({file_path: file_content})

        commit = commit_path[0]
        new_commit_path = commit_path[1]

        return commit.apply(CommitIndex._restore_file_recursive(file_path, file_content, new_commit_path))


def init_fresh_repo(repo):
    repo.init()

    c1 = Commit(str(uuid4()), 'c1', None, [
        Change('a.txt', ChangeType.ADD, [
            Operation(OperationType.INSERT, (0, 0), 'Hello world!')],
               ),
        Change('b.txt', ChangeType.ADD, [
            Operation(OperationType.INSERT, (0, 0), 'Second file')],
               ),
    ])
    c2 = Commit(str(uuid4()), 'c2', [c1.id], [
        Change('a.txt', ChangeType.EDIT, [
            Operation(OperationType.REPLACE, (6, 12), 'Krystofee!'),
            Operation(OperationType.INSERT, (12, 12), ' How are you?'),
        ]),
    ])
    c3 = Commit(str(uuid4()), 'c3', [c2.id], [
        Change('b.txt', ChangeType.REMOVE, [
            Operation(OperationType.INSERT, (0, 0), 'Hello world!')],
               ),
        Change('test/x.txt', ChangeType.ADD, [
            Operation(OperationType.INSERT, (0, 0), 'test nested file')],
               ),
    ])
    c4 = Commit(str(uuid4()), 'c4', [c3.id], [
        Change('a.txt', ChangeType.EDIT, [
            Operation(OperationType.INSERT, (0, 0), '... ')],
               ),
        Change('b.txt', ChangeType.ADD, [
            Operation(OperationType.INSERT, (0, 0), 'NEW B!!!')],
               ),
    ])

    storage = CommitStorage(repo)
    storage.store(c1)
    storage.store(c2)
    storage.store(c3)
    storage.store(c4)
    storage.save()

    repo._update_head_ref(c4.id)

def print_indexed_files(commit_index):
    print('Indexed files:')
    print(commit_index.indexed_files)
    print()

    for indexed_file in commit_index.indexed_files.keys():
        print('restored', indexed_file, 'is', commit_index.restore_file(indexed_file))


def get_operations(original, new):
    matcher = SequenceMatcher(lambda x: x == " ", original, new)

    operations = []
    for opcode in matcher.get_opcodes():
        operation_type = OperationType.get_type(opcode[0])
        if operation_type:
            operations.append(
                Operation(
                    OperationType.get_type(opcode[0]), (opcode[1], opcode[2]), new[opcode[3]:opcode[4]]
                )
            )
    return operations


def create_commit(storage, work_index, against_commit, message):
    if against_commit == None:
        pass # Test

    changes = []

    # tracked_files = set(commit_index.indexed_files.keys())
    staged_files = work_index.staged

    visited_files = set()

    if against_commit:
        commit_index = CommitIndex(storage, against_commit)
        commit_index.build()

        for file_path in commit_index.indexed_files.keys():
            visited_files.add(file_path)
            if os.path.exists(file_path):
                with open(file_path) as file:
                    original_content = commit_index.restore_file(file_path)
                    current_content = ''.join(file.readlines())
                    operations = get_operations(original_content, current_content)
                    if operations:
                        changes.append(Change(file_path, ChangeType.EDIT, operations))
            else:
                changes.append(Change(file_path, ChangeType.REMOVE, []))

    for file_path in staged_files - set(visited_files):
        with open(file_path) as file:
            current_content = ''.join(file.readlines())
            changes.append(Change(file_path, ChangeType.ADD, [Operation(OperationType.INSERT, (0, 0), current_content)]))

    parents = []
    if against_commit:
        parents.append(against_commit)
    else:
        parents.append(INIT_COMMIT_PARENT)

    return Commit(str(uuid4()), message, parents, changes)

# repo = Repository()
# init_fresh_repo(repo)
# repo.load()
#
# commit_index = CommitIndex(repo.storage, repo.head_ref)
# commit_index.build()
#
# work_index = WorkIndex.load(repo)
#
# changes = create_commit(repo.storage, work_index, repo.head_ref)
# commit = Commit(str(uuid4()), [repo.head_ref], changes)
#
# print(commit)
# print(commit.changes)

# repo.create_commit(commit)


@click.group()
def kit_cli():
    pass


@click.command()
def init():
    print('Initializing repository in .')
    repo = Repository()
    if repo.exists():
        print('... repository already exists')
    else:
        repo.init()
        print('... initialized repository at .')


@click.command()
@click.argument('file_path', required=True)
def add(file_path):
    repo = Repository()
    repo.load()
    repo.work_index.stage(file_path)
    repo.work_index.save()


@click.command()
@click.option('--message', required=True, help="Attach commit message")
def commit(message):
    repo = Repository()
    repo.load()

    commit = create_commit(repo.storage, repo.work_index, repo.head_ref, message)

    print('...created commit')
    print()
    print(commit.to_string())

    repo.create_commit(commit)


@click.command()
def log():
    repo = Repository()
    repo.load()

    commit_index = CommitIndex(repo.storage, repo.head_ref)
    history = commit_index.get_commit_path(repo.head_ref, INIT_COMMIT_PARENT)

    if not len(history):
        print('Nothing to display.')

    while history:
        commit = history[0]
        print(commit.to_string())

        if len(history) <= 1:
            break

        history = history[1]


@click.command()
def status():
    repo = Repository()
    repo.load()

    if repo.work_index.staged:
        print('Staged files:')
        for file in repo.work_index.staged:
            print('- ', file)
    else:
        print('Nothing staged...')

    if repo.work_index.tracked:
        print('Tracked files:')
        for file in repo.work_index.tracked:
            print('- ', file)
    else:
        print('Nothing tracked...')


@click.command()
@click.option('--file', required=True, help="Path to the file")
@click.option('--commit', required=True, help="Commit ID")
def cat(file, commit):
    repo = Repository()
    repo.load()

    commit_index = CommitIndex(repo.storage, commit)
    commit_index.build()

    restored_file = commit_index.restore_file(file)

    print(f'Version {commit} of file {file} is:')
    print('------ SOF ------')
    print(restored_file)
    print('------ EOF ------')


kit_cli.add_command(init)
kit_cli.add_command(add)
kit_cli.add_command(commit)
kit_cli.add_command(log)
kit_cli.add_command(status)
kit_cli.add_command(cat)


if __name__ == '__main__':
    kit_cli()

        print('Nothing staged...')

    if repo.work_index.tracked:
        print('Tracked files:')
        for file in repo.work_index.tracked:
            print('- ', file)
    else:
        print('Nothing tracked...')


@click.command()
@click.option('--file', required=True, help="Path to the file")
@click.option('--commit', required=True, help="Commit ID")
def cat(file, commit):
    repo = Repository()
    repo.load()

    commit_index = CommitIndex(repo.storage, commit)
    commit_index.build()

    restored_file = commit_index.restore_file(file)

    print(f'Version {commit} of file {file} is:')
    print('------ SOF ------')
    print(restored_file)
    print('------ EOF ------')


kit_cli.add_command(init)
kit_cli.add_command(add)
kit_cli.add_command(commit)
kit_cli.add_command(log)
kit_cli.add_command(status)
kit_cli.add_command(cat)


if __name__ == '__main__':
    kit_cli()
