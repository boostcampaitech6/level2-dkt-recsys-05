"use strict";
function requireWithFallback(electronModule, nodeModule) { try { return require(electronModule); } catch (err) {} return require(nodeModule); }
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.DataSource = exports.GIT_CONFIG = void 0;
const cp = require("child_process");
const fs = requireWithFallback("original-fs", "fs");
const iconv_lite_1 = require("iconv-lite");
const path = require("path");
const vscode = require("vscode");
const askpassManager_1 = require("./askpass/askpassManager");
const config_1 = require("./config");
const utils_1 = require("./utils");
const disposable_1 = require("./utils/disposable");
const DRIVE_LETTER_PATH_REGEX = /^[a-z]:\//;
const EOL_REGEX = /\r\n|\r|\n/g;
const INVALID_BRANCH_REGEXP = /^\(.* .*\)$/;
const REMOTE_HEAD_BRANCH_REGEXP = /^remotes\/.*\/HEAD$/;
const GIT_LOG_SEPARATOR = 'XX7Nal-YARtTpjCikii9nJxER19D6diSyk-AWkPb';
exports.GIT_CONFIG = {
    DIFF: {
        GUI_TOOL: 'diff.guitool',
        TOOL: 'diff.tool'
    },
    REMOTE: {
        PUSH_DEFAULT: 'remote.pushdefault'
    },
    USER: {
        EMAIL: 'user.email',
        NAME: 'user.name'
    }
};
class DataSource extends disposable_1.Disposable {
    constructor(gitExecutable, onDidChangeConfiguration, onDidChangeGitExecutable, logger) {
        super();
        this.logger = logger;
        this.setGitExecutable(gitExecutable);
        const askpassManager = new askpassManager_1.AskpassManager();
        this.askpassEnv = askpassManager.getEnv();
        this.registerDisposables(onDidChangeConfiguration((event) => {
            if (event.affectsConfiguration('git-graph.date.type') || event.affectsConfiguration('git-graph.dateType') ||
                event.affectsConfiguration('git-graph.repository.commits.showSignatureStatus') || event.affectsConfiguration('git-graph.showSignatureStatus') ||
                event.affectsConfiguration('git-graph.repository.useMailmap') || event.affectsConfiguration('git-graph.useMailmap')) {
                this.generateGitCommandFormats();
            }
        }), onDidChangeGitExecutable((gitExecutable) => {
            this.setGitExecutable(gitExecutable);
        }), askpassManager);
    }
    isGitExecutableUnknown() {
        return this.gitExecutable === null;
    }
    setGitExecutable(gitExecutable) {
        this.gitExecutable = gitExecutable;
        this.gitExecutableSupportsGpgInfo = gitExecutable !== null ? utils_1.doesVersionMeetRequirement(gitExecutable.version, '2.4.0') : false;
        this.generateGitCommandFormats();
    }
    generateGitCommandFormats() {
        const config = config_1.getConfig();
        const dateType = config.dateType === 0 ? '%at' : '%ct';
        const useMailmap = config.useMailmap;
        this.gitFormatCommitDetails = [
            '%H', '%P',
            useMailmap ? '%aN' : '%an', useMailmap ? '%aE' : '%ae', '%at', useMailmap ? '%cN' : '%cn', useMailmap ? '%cE' : '%ce', '%ct',
            ...(config.showSignatureStatus && this.gitExecutableSupportsGpgInfo ? ['%G?', '%GS', '%GK'] : ['', '', '']),
            '%B'
        ].join(GIT_LOG_SEPARATOR);
        this.gitFormatLog = [
            '%H', '%P',
            useMailmap ? '%aN' : '%an', useMailmap ? '%aE' : '%ae', dateType,
            '%s'
        ].join(GIT_LOG_SEPARATOR);
        this.gitFormatStash = [
            '%H', '%P', '%gD',
            useMailmap ? '%aN' : '%an', useMailmap ? '%aE' : '%ae', dateType,
            '%s'
        ].join(GIT_LOG_SEPARATOR);
    }
    getRepoInfo(repo, showRemoteBranches, showStashes, hideRemotes) {
        return Promise.all([
            this.getBranches(repo, showRemoteBranches, hideRemotes),
            this.getRemotes(repo),
            showStashes ? this.getStashes(repo) : Promise.resolve([])
        ]).then((results) => {
            return { branches: results[0].branches, head: results[0].head, remotes: results[1], stashes: results[2], error: null };
        }).catch((errorMessage) => {
            return { branches: [], head: null, remotes: [], stashes: [], error: errorMessage };
        });
    }
    getCommits(repo, branches, maxCommits, showTags, showRemoteBranches, includeCommitsMentionedByReflogs, onlyFollowFirstParent, commitOrdering, remotes, hideRemotes, stashes) {
        const config = config_1.getConfig();
        return Promise.all([
            this.getLog(repo, branches, maxCommits + 1, showTags && config.showCommitsOnlyReferencedByTags, showRemoteBranches, includeCommitsMentionedByReflogs, onlyFollowFirstParent, commitOrdering, remotes, hideRemotes, stashes),
            this.getRefs(repo, showRemoteBranches, config.showRemoteHeads, hideRemotes).then((refData) => refData, (errorMessage) => errorMessage)
        ]).then((results) => __awaiter(this, void 0, void 0, function* () {
            let commits = results[0], refData = results[1], i;
            let moreCommitsAvailable = commits.length === maxCommits + 1;
            if (moreCommitsAvailable)
                commits.pop();
            if (typeof refData === 'string') {
                if (commits.length > 0) {
                    throw refData;
                }
                else {
                    refData = { head: null, heads: [], tags: [], remotes: [] };
                }
            }
            if (refData.head !== null && config.showUncommittedChanges) {
                for (i = 0; i < commits.length; i++) {
                    if (refData.head === commits[i].hash) {
                        const numUncommittedChanges = yield this.getUncommittedChanges(repo);
                        if (numUncommittedChanges > 0) {
                            commits.unshift({ hash: utils_1.UNCOMMITTED, parents: [refData.head], author: '*', email: '', date: Math.round((new Date()).getTime() / 1000), message: 'Uncommitted Changes (' + numUncommittedChanges + ')' });
                        }
                        break;
                    }
                }
            }
            let commitNodes = [];
            let commitLookup = {};
            for (i = 0; i < commits.length; i++) {
                commitLookup[commits[i].hash] = i;
                commitNodes.push(Object.assign(Object.assign({}, commits[i]), { heads: [], tags: [], remotes: [], stash: null }));
            }
            let toAdd = [];
            for (i = 0; i < stashes.length; i++) {
                if (typeof commitLookup[stashes[i].hash] === 'number') {
                    commitNodes[commitLookup[stashes[i].hash]].stash = {
                        selector: stashes[i].selector,
                        baseHash: stashes[i].baseHash,
                        untrackedFilesHash: stashes[i].untrackedFilesHash
                    };
                }
                else if (typeof commitLookup[stashes[i].baseHash] === 'number') {
                    toAdd.push({ index: commitLookup[stashes[i].baseHash], data: stashes[i] });
                }
            }
            toAdd.sort((a, b) => a.index !== b.index ? a.index - b.index : b.data.date - a.data.date);
            for (i = toAdd.length - 1; i >= 0; i--) {
                let stash = toAdd[i].data;
                commitNodes.splice(toAdd[i].index, 0, {
                    hash: stash.hash,
                    parents: [stash.baseHash],
                    author: stash.author,
                    email: stash.email,
                    date: stash.date,
                    message: stash.message,
                    heads: [], tags: [], remotes: [],
                    stash: {
                        selector: stash.selector,
                        baseHash: stash.baseHash,
                        untrackedFilesHash: stash.untrackedFilesHash
                    }
                });
            }
            for (i = 0; i < commitNodes.length; i++) {
                commitLookup[commitNodes[i].hash] = i;
            }
            for (i = 0; i < refData.heads.length; i++) {
                if (typeof commitLookup[refData.heads[i].hash] === 'number')
                    commitNodes[commitLookup[refData.heads[i].hash]].heads.push(refData.heads[i].name);
            }
            if (showTags) {
                for (i = 0; i < refData.tags.length; i++) {
                    if (typeof commitLookup[refData.tags[i].hash] === 'number')
                        commitNodes[commitLookup[refData.tags[i].hash]].tags.push({ name: refData.tags[i].name, annotated: refData.tags[i].annotated });
                }
            }
            for (i = 0; i < refData.remotes.length; i++) {
                if (typeof commitLookup[refData.remotes[i].hash] === 'number') {
                    let name = refData.remotes[i].name;
                    let remote = remotes.find(remote => name.startsWith(remote + '/'));
                    commitNodes[commitLookup[refData.remotes[i].hash]].remotes.push({ name: name, remote: remote ? remote : null });
                }
            }
            return {
                commits: commitNodes,
                head: refData.head,
                tags: unique(refData.tags.map((tag) => tag.name)),
                moreCommitsAvailable: moreCommitsAvailable,
                error: null
            };
        })).catch((errorMessage) => {
            return { commits: [], head: null, tags: [], moreCommitsAvailable: false, error: errorMessage };
        });
    }
    getConfig(repo, remotes) {
        return Promise.all([
            this.getConfigList(repo),
            this.getConfigList(repo, "local"),
            this.getConfigList(repo, "global")
        ]).then((results) => {
            const consolidatedConfigs = results[0], localConfigs = results[1], globalConfigs = results[2];
            const branches = {};
            Object.keys(localConfigs).forEach((key) => {
                if (key.startsWith('branch.')) {
                    if (key.endsWith('.remote')) {
                        const branchName = key.substring(7, key.length - 7);
                        branches[branchName] = {
                            pushRemote: typeof branches[branchName] !== 'undefined' ? branches[branchName].pushRemote : null,
                            remote: localConfigs[key]
                        };
                    }
                    else if (key.endsWith('.pushremote')) {
                        const branchName = key.substring(7, key.length - 11);
                        branches[branchName] = {
                            pushRemote: localConfigs[key],
                            remote: typeof branches[branchName] !== 'undefined' ? branches[branchName].remote : null
                        };
                    }
                }
            });
            return {
                config: {
                    branches: branches,
                    diffTool: getConfigValue(consolidatedConfigs, exports.GIT_CONFIG.DIFF.TOOL),
                    guiDiffTool: getConfigValue(consolidatedConfigs, exports.GIT_CONFIG.DIFF.GUI_TOOL),
                    pushDefault: getConfigValue(consolidatedConfigs, exports.GIT_CONFIG.REMOTE.PUSH_DEFAULT),
                    remotes: remotes.map((remote) => ({
                        name: remote,
                        url: getConfigValue(localConfigs, 'remote.' + remote + '.url'),
                        pushUrl: getConfigValue(localConfigs, 'remote.' + remote + '.pushurl')
                    })),
                    user: {
                        name: {
                            local: getConfigValue(localConfigs, exports.GIT_CONFIG.USER.NAME),
                            global: getConfigValue(globalConfigs, exports.GIT_CONFIG.USER.NAME)
                        },
                        email: {
                            local: getConfigValue(localConfigs, exports.GIT_CONFIG.USER.EMAIL),
                            global: getConfigValue(globalConfigs, exports.GIT_CONFIG.USER.EMAIL)
                        }
                    }
                },
                error: null
            };
        }).catch((errorMessage) => {
            return { config: null, error: errorMessage };
        });
    }
    getCommitDetails(repo, commitHash, hasParents) {
        const fromCommit = commitHash + (hasParents ? '^' : '');
        return Promise.all([
            this.getCommitDetailsBase(repo, commitHash),
            this.getDiffNameStatus(repo, fromCommit, commitHash),
            this.getDiffNumStat(repo, fromCommit, commitHash)
        ]).then((results) => {
            results[0].fileChanges = generateFileChanges(results[1], results[2], null);
            return { commitDetails: results[0], error: null };
        }).catch((errorMessage) => {
            return { commitDetails: null, error: errorMessage };
        });
    }
    getStashDetails(repo, commitHash, stash) {
        return Promise.all([
            this.getCommitDetailsBase(repo, commitHash),
            this.getDiffNameStatus(repo, stash.baseHash, commitHash),
            this.getDiffNumStat(repo, stash.baseHash, commitHash),
            stash.untrackedFilesHash !== null ? this.getDiffNameStatus(repo, stash.untrackedFilesHash, stash.untrackedFilesHash) : Promise.resolve([]),
            stash.untrackedFilesHash !== null ? this.getDiffNumStat(repo, stash.untrackedFilesHash, stash.untrackedFilesHash) : Promise.resolve([])
        ]).then((results) => {
            results[0].fileChanges = generateFileChanges(results[1], results[2], null);
            if (stash.untrackedFilesHash !== null) {
                generateFileChanges(results[3], results[4], null).forEach((fileChange) => {
                    if (fileChange.type === "A") {
                        fileChange.type = "U";
                        results[0].fileChanges.push(fileChange);
                    }
                });
            }
            return { commitDetails: results[0], error: null };
        }).catch((errorMessage) => {
            return { commitDetails: null, error: errorMessage };
        });
    }
    getUncommittedDetails(repo) {
        return Promise.all([
            this.getDiffNameStatus(repo, 'HEAD', ''),
            this.getDiffNumStat(repo, 'HEAD', ''),
            this.getStatus(repo)
        ]).then((results) => {
            return {
                commitDetails: {
                    hash: utils_1.UNCOMMITTED, parents: [],
                    author: '', authorEmail: '', authorDate: 0,
                    committer: '', committerEmail: '', committerDate: 0, signature: null,
                    body: '', fileChanges: generateFileChanges(results[0], results[1], results[2])
                },
                error: null
            };
        }).catch((errorMessage) => {
            return { commitDetails: null, error: errorMessage };
        });
    }
    getCommitComparison(repo, fromHash, toHash) {
        return Promise.all([
            this.getDiffNameStatus(repo, fromHash, toHash === utils_1.UNCOMMITTED ? '' : toHash),
            this.getDiffNumStat(repo, fromHash, toHash === utils_1.UNCOMMITTED ? '' : toHash),
            toHash === utils_1.UNCOMMITTED ? this.getStatus(repo) : Promise.resolve(null)
        ]).then((results) => {
            return {
                fileChanges: generateFileChanges(results[0], results[1], results[2]),
                error: null
            };
        }).catch((errorMessage) => {
            return { fileChanges: [], error: errorMessage };
        });
    }
    getCommitFile(repo, commitHash, filePath) {
        return this._spawnGit(['show', commitHash + ':' + filePath], repo, stdout => {
            const encoding = config_1.getConfig(repo).fileEncoding;
            return iconv_lite_1.decode(stdout, iconv_lite_1.encodingExists(encoding) ? encoding : 'utf8');
        });
    }
    getCommitSubject(repo, commitHash) {
        return this.spawnGit(['-c', 'log.showSignature=false', 'log', '--format=%s', '-n', '1', commitHash, '--'], repo, (stdout) => {
            return stdout.trim().replace(/\s+/g, ' ');
        }).then((subject) => subject, () => null);
    }
    getRemoteUrl(repo, remote) {
        return this.spawnGit(['config', '--get', 'remote.' + remote + '.url'], repo, (stdout) => {
            return stdout.split(EOL_REGEX)[0];
        }).then((url) => url, () => null);
    }
    getNewPathOfRenamedFile(repo, commitHash, oldFilePath) {
        return this.getDiffNameStatus(repo, commitHash, '', 'R').then((renamed) => {
            const renamedRecordForFile = renamed.find((record) => record.oldFilePath === oldFilePath);
            return renamedRecordForFile ? renamedRecordForFile.newFilePath : null;
        }).catch(() => null);
    }
    getTagDetails(repo, tagName) {
        return this.spawnGit(['for-each-ref', 'refs/tags/' + tagName, '--format=' + ['%(objectname)', '%(taggername)', '%(taggeremail)', '%(taggerdate:unix)', '%(contents)'].join(GIT_LOG_SEPARATOR)], repo, (stdout) => {
            let data = stdout.split(GIT_LOG_SEPARATOR);
            return {
                tagHash: data[0],
                name: data[1],
                email: data[2].substring(data[2].startsWith('<') ? 1 : 0, data[2].length - (data[2].endsWith('>') ? 1 : 0)),
                date: parseInt(data[3]),
                message: removeTrailingBlankLines(data[4].split(EOL_REGEX)).join('\n'),
                error: null
            };
        }).then((data) => {
            return data;
        }).catch((errorMessage) => {
            return { tagHash: '', name: '', email: '', date: 0, message: '', error: errorMessage };
        });
    }
    getSubmodules(repo) {
        return new Promise(resolve => {
            fs.readFile(path.join(repo, '.gitmodules'), { encoding: 'utf8' }, (err, data) => __awaiter(this, void 0, void 0, function* () {
                let submodules = [];
                if (!err) {
                    let lines = data.split(EOL_REGEX), inSubmoduleSection = false, match;
                    const section = /^\s*\[.*\]\s*$/, submodule = /^\s*\[submodule "([^"]+)"\]\s*$/, pathProp = /^\s*path\s+=\s+(.*)$/;
                    for (let i = 0; i < lines.length; i++) {
                        if (lines[i].match(section) !== null) {
                            inSubmoduleSection = lines[i].match(submodule) !== null;
                            continue;
                        }
                        if (inSubmoduleSection && (match = lines[i].match(pathProp)) !== null) {
                            let root = yield this.repoRoot(utils_1.getPathFromUri(vscode.Uri.file(path.join(repo, utils_1.getPathFromStr(match[1])))));
                            if (root !== null && !submodules.includes(root)) {
                                submodules.push(root);
                            }
                        }
                    }
                }
                resolve(submodules);
            }));
        });
    }
    areStagedChanges(repo) {
        return this.spawnGit(['diff-index', 'HEAD'], repo, (stdout) => stdout !== '').then(changes => changes, () => false);
    }
    repoRoot(pathOfPotentialRepo) {
        return this.spawnGit(['rev-parse', '--show-toplevel'], pathOfPotentialRepo, (stdout) => utils_1.getPathFromUri(vscode.Uri.file(path.normalize(stdout.trim())))).then((pathReturnedByGit) => __awaiter(this, void 0, void 0, function* () {
            if (process.platform === 'win32') {
                let driveLetterPathMatch;
                if ((driveLetterPathMatch = pathOfPotentialRepo.match(DRIVE_LETTER_PATH_REGEX)) && !pathReturnedByGit.match(DRIVE_LETTER_PATH_REGEX)) {
                    const realPathForDriveLetter = utils_1.pathWithTrailingSlash(yield utils_1.realpath(driveLetterPathMatch[0], true));
                    if (realPathForDriveLetter !== driveLetterPathMatch[0] && pathReturnedByGit.startsWith(realPathForDriveLetter)) {
                        pathReturnedByGit = driveLetterPathMatch[0] + pathReturnedByGit.substring(realPathForDriveLetter.length);
                    }
                }
            }
            let path = pathOfPotentialRepo;
            let first = path.indexOf('/');
            while (true) {
                if (pathReturnedByGit === path || pathReturnedByGit === (yield utils_1.realpath(path)))
                    return path;
                let next = path.lastIndexOf('/');
                if (first !== next && next > -1) {
                    path = path.substring(0, next);
                }
                else {
                    return pathReturnedByGit;
                }
            }
        })).catch(() => null);
    }
    addRemote(repo, name, url, pushUrl, fetch) {
        return __awaiter(this, void 0, void 0, function* () {
            let status = yield this.runGitCommand(['remote', 'add', name, url], repo);
            if (status !== null)
                return status;
            if (pushUrl !== null) {
                status = yield this.runGitCommand(['remote', 'set-url', name, '--push', pushUrl], repo);
                if (status !== null)
                    return status;
            }
            return fetch ? this.fetch(repo, name, false, false) : null;
        });
    }
    deleteRemote(repo, name) {
        return this.runGitCommand(['remote', 'remove', name], repo);
    }
    editRemote(repo, nameOld, nameNew, urlOld, urlNew, pushUrlOld, pushUrlNew) {
        return __awaiter(this, void 0, void 0, function* () {
            if (nameOld !== nameNew) {
                let status = yield this.runGitCommand(['remote', 'rename', nameOld, nameNew], repo);
                if (status !== null)
                    return status;
            }
            if (urlOld !== urlNew) {
                let args = ['remote', 'set-url', nameNew];
                if (urlNew === null)
                    args.push('--delete', urlOld);
                else if (urlOld === null)
                    args.push('--add', urlNew);
                else
                    args.push(urlNew, urlOld);
                let status = yield this.runGitCommand(args, repo);
                if (status !== null)
                    return status;
            }
            if (pushUrlOld !== pushUrlNew) {
                let args = ['remote', 'set-url', '--push', nameNew];
                if (pushUrlNew === null)
                    args.push('--delete', pushUrlOld);
                else if (pushUrlOld === null)
                    args.push('--add', pushUrlNew);
                else
                    args.push(pushUrlNew, pushUrlOld);
                let status = yield this.runGitCommand(args, repo);
                if (status !== null)
                    return status;
            }
            return null;
        });
    }
    pruneRemote(repo, name) {
        return this.runGitCommand(['remote', 'prune', name], repo);
    }
    addTag(repo, tagName, commitHash, type, message, force) {
        const args = ['tag'];
        if (force) {
            args.push('-f');
        }
        if (type === 1) {
            args.push(tagName);
        }
        else {
            args.push(config_1.getConfig().signTags ? '-s' : '-a', tagName, '-m', message);
        }
        args.push(commitHash);
        return this.runGitCommand(args, repo);
    }
    deleteTag(repo, tagName, deleteOnRemote) {
        return __awaiter(this, void 0, void 0, function* () {
            if (deleteOnRemote !== null) {
                let status = yield this.runGitCommand(['push', deleteOnRemote, '--delete', tagName], repo);
                if (status !== null)
                    return status;
            }
            return this.runGitCommand(['tag', '-d', tagName], repo);
        });
    }
    fetch(repo, remote, prune, pruneTags) {
        let args = ['fetch', remote === null ? '--all' : remote];
        if (prune) {
            args.push('--prune');
        }
        if (pruneTags) {
            if (!prune) {
                return Promise.resolve('In order to Prune Tags, pruning must also be enabled when fetching from ' + (remote !== null ? 'a remote' : 'remote(s)') + '.');
            }
            else if (this.gitExecutable !== null && !utils_1.doesVersionMeetRequirement(this.gitExecutable.version, '2.17.0')) {
                return Promise.resolve(utils_1.constructIncompatibleGitVersionMessage(this.gitExecutable, '2.17.0', 'pruning tags when fetching'));
            }
            args.push('--prune-tags');
        }
        return this.runGitCommand(args, repo);
    }
    pushBranch(repo, branchName, remote, setUpstream, mode) {
        let args = ['push'];
        args.push(remote, branchName);
        if (setUpstream)
            args.push('--set-upstream');
        if (mode !== "")
            args.push('--' + mode);
        return this.runGitCommand(args, repo);
    }
    pushTag(repo, tagName, remote) {
        return this.runGitCommand(['push', remote, tagName], repo);
    }
    pushBranchToMultipleRemotes(repo, branchName, remotes, setUpstream, mode) {
        return __awaiter(this, void 0, void 0, function* () {
            if (remotes.length === 0) {
                return ['No remote(s) were specified to push the branch ' + branchName + ' to.'];
            }
            const results = [];
            for (let i = 0; i < remotes.length; i++) {
                const result = yield this.pushBranch(repo, branchName, remotes[i], setUpstream, mode);
                results.push(result);
                if (result !== null)
                    break;
            }
            return results;
        });
    }
    pushTagToMultipleRemotes(repo, tagName, remotes) {
        return __awaiter(this, void 0, void 0, function* () {
            if (remotes.length === 0) {
                return ['No remote(s) were specified to push the tag ' + tagName + ' to.'];
            }
            const results = [];
            for (let i = 0; i < remotes.length; i++) {
                const result = yield this.pushTag(repo, tagName, remotes[i]);
                results.push(result);
                if (result !== null)
                    break;
            }
            return results;
        });
    }
    checkoutBranch(repo, branchName, remoteBranch) {
        let args = ['checkout'];
        if (remoteBranch === null)
            args.push(branchName);
        else
            args.push('-b', branchName, remoteBranch);
        return this.runGitCommand(args, repo);
    }
    createBranch(repo, branchName, commitHash, checkout, force) {
        return __awaiter(this, void 0, void 0, function* () {
            const args = [];
            if (checkout && !force) {
                args.push('checkout', '-b');
            }
            else {
                args.push('branch');
                if (force) {
                    args.push('-f');
                }
            }
            args.push(branchName, commitHash);
            const statuses = [yield this.runGitCommand(args, repo)];
            if (statuses[0] === null && checkout && force) {
                statuses.push(yield this.checkoutBranch(repo, branchName, null));
            }
            return statuses;
        });
    }
    deleteBranch(repo, branchName, forceDelete) {
        let args = ['branch', '--delete'];
        if (forceDelete)
            args.push('--force');
        args.push(branchName);
        return this.runGitCommand(args, repo);
    }
    deleteRemoteBranch(repo, branchName, remote) {
        return __awaiter(this, void 0, void 0, function* () {
            let remoteStatus = yield this.runGitCommand(['push', remote, '--delete', branchName], repo);
            if (remoteStatus !== null && (new RegExp('remote ref does not exist', 'i')).test(remoteStatus)) {
                let trackingBranchStatus = yield this.runGitCommand(['branch', '-d', '-r', remote + '/' + branchName], repo);
                return trackingBranchStatus === null ? null : 'Branch does not exist on the remote, deleting the remote tracking branch ' + remote + '/' + branchName + '.\n' + trackingBranchStatus;
            }
            return remoteStatus;
        });
    }
    fetchIntoLocalBranch(repo, remote, remoteBranch, localBranch, force) {
        const args = ['fetch'];
        if (force) {
            args.push('-f');
        }
        args.push(remote, remoteBranch + ':' + localBranch);
        return this.runGitCommand(args, repo);
    }
    pullBranch(repo, branchName, remote, createNewCommit, squash) {
        const args = ['pull', remote, branchName], config = config_1.getConfig();
        if (squash) {
            args.push('--squash');
        }
        else if (createNewCommit) {
            args.push('--no-ff');
        }
        if (config.signCommits) {
            args.push('-S');
        }
        return this.runGitCommand(args, repo).then((pullStatus) => {
            return pullStatus === null && squash
                ? this.commitSquashIfStagedChangesExist(repo, remote + '/' + branchName, "Branch", config.squashPullMessageFormat, config.signCommits)
                : pullStatus;
        });
    }
    renameBranch(repo, oldName, newName) {
        return this.runGitCommand(['branch', '-m', oldName, newName], repo);
    }
    merge(repo, obj, actionOn, createNewCommit, squash, noCommit) {
        const args = ['merge', obj], config = config_1.getConfig();
        if (squash) {
            args.push('--squash');
        }
        else if (createNewCommit) {
            args.push('--no-ff');
        }
        if (noCommit) {
            args.push('--no-commit');
        }
        if (config.signCommits) {
            args.push('-S');
        }
        return this.runGitCommand(args, repo).then((mergeStatus) => {
            return mergeStatus === null && squash && !noCommit
                ? this.commitSquashIfStagedChangesExist(repo, obj, actionOn, config.squashMergeMessageFormat, config.signCommits)
                : mergeStatus;
        });
    }
    rebase(repo, obj, actionOn, ignoreDate, interactive) {
        if (interactive) {
            return this.openGitTerminal(repo, 'rebase --interactive ' + (config_1.getConfig().signCommits ? '-S ' : '') + (actionOn === "Branch" ? obj.replace(/'/g, '"\'"') : obj), 'Rebase on "' + (actionOn === "Branch" ? obj : utils_1.abbrevCommit(obj)) + '"');
        }
        else {
            const args = ['rebase', obj];
            if (ignoreDate) {
                args.push('--ignore-date');
            }
            if (config_1.getConfig().signCommits) {
                args.push('-S');
            }
            return this.runGitCommand(args, repo);
        }
    }
    archive(repo, ref, outputFilePath, type) {
        return this.runGitCommand(['archive', '--format=' + type, '-o', outputFilePath, ref], repo);
    }
    checkoutCommit(repo, commitHash) {
        return this.runGitCommand(['checkout', commitHash], repo);
    }
    cherrypickCommit(repo, commitHash, parentIndex, recordOrigin, noCommit) {
        const args = ['cherry-pick'];
        if (noCommit) {
            args.push('--no-commit');
        }
        if (recordOrigin) {
            args.push('-x');
        }
        if (config_1.getConfig().signCommits) {
            args.push('-S');
        }
        if (parentIndex > 0) {
            args.push('-m', parentIndex.toString());
        }
        args.push(commitHash);
        return this.runGitCommand(args, repo);
    }
    dropCommit(repo, commitHash) {
        const args = ['rebase'];
        if (config_1.getConfig().signCommits) {
            args.push('-S');
        }
        args.push('--onto', commitHash + '^', commitHash);
        return this.runGitCommand(args, repo);
    }
    resetToCommit(repo, commit, resetMode) {
        return this.runGitCommand(['reset', '--' + resetMode, commit], repo);
    }
    revertCommit(repo, commitHash, parentIndex) {
        const args = ['revert', '--no-edit'];
        if (config_1.getConfig().signCommits) {
            args.push('-S');
        }
        if (parentIndex > 0) {
            args.push('-m', parentIndex.toString());
        }
        args.push(commitHash);
        return this.runGitCommand(args, repo);
    }
    setConfigValue(repo, key, value, location) {
        return this.runGitCommand(['config', '--' + location, key, value], repo);
    }
    unsetConfigValue(repo, key, location) {
        return this.runGitCommand(['config', '--' + location, '--unset-all', key], repo);
    }
    cleanUntrackedFiles(repo, directories) {
        return this.runGitCommand(['clean', '-f' + (directories ? 'd' : '')], repo);
    }
    applyStash(repo, selector, reinstateIndex) {
        let args = ['stash', 'apply'];
        if (reinstateIndex)
            args.push('--index');
        args.push(selector);
        return this.runGitCommand(args, repo);
    }
    branchFromStash(repo, selector, branchName) {
        return this.runGitCommand(['stash', 'branch', branchName, selector], repo);
    }
    dropStash(repo, selector) {
        return this.runGitCommand(['stash', 'drop', selector], repo);
    }
    popStash(repo, selector, reinstateIndex) {
        let args = ['stash', 'pop'];
        if (reinstateIndex)
            args.push('--index');
        args.push(selector);
        return this.runGitCommand(args, repo);
    }
    pushStash(repo, message, includeUntracked) {
        if (this.gitExecutable === null) {
            return Promise.resolve(utils_1.UNABLE_TO_FIND_GIT_MSG);
        }
        else if (!utils_1.doesVersionMeetRequirement(this.gitExecutable.version, '2.13.2')) {
            return Promise.resolve(utils_1.constructIncompatibleGitVersionMessage(this.gitExecutable, '2.13.2'));
        }
        let args = ['stash', 'push'];
        if (includeUntracked)
            args.push('--include-untracked');
        if (message !== '')
            args.push('--message', message);
        return this.runGitCommand(args, repo);
    }
    openExternalDirDiff(repo, fromHash, toHash, isGui) {
        return new Promise((resolve) => {
            if (this.gitExecutable === null) {
                resolve(utils_1.UNABLE_TO_FIND_GIT_MSG);
            }
            else {
                const args = ['difftool', '--dir-diff'];
                if (isGui) {
                    args.push('-g');
                }
                if (fromHash === toHash) {
                    if (toHash === utils_1.UNCOMMITTED) {
                        args.push('HEAD');
                    }
                    else {
                        args.push(toHash + '^..' + toHash);
                    }
                }
                else {
                    if (toHash === utils_1.UNCOMMITTED) {
                        args.push(fromHash);
                    }
                    else {
                        args.push(fromHash + '..' + toHash);
                    }
                }
                if (isGui) {
                    this.logger.log('External diff tool is being opened (' + args[args.length - 1] + ')');
                    this.runGitCommand(args, repo).then((errorInfo) => {
                        this.logger.log('External diff tool has exited (' + args[args.length - 1] + ')');
                        if (errorInfo !== null) {
                            const errorMessage = errorInfo.replace(EOL_REGEX, ' ');
                            this.logger.logError(errorMessage);
                            utils_1.showErrorMessage(errorMessage);
                        }
                    });
                }
                else {
                    utils_1.openGitTerminal(repo, this.gitExecutable.path, args.join(' '), 'Open External Directory Diff');
                }
                setTimeout(() => resolve(null), 1500);
            }
        });
    }
    openGitTerminal(repo, command, name) {
        return new Promise((resolve) => {
            if (this.gitExecutable === null) {
                resolve(utils_1.UNABLE_TO_FIND_GIT_MSG);
            }
            else {
                utils_1.openGitTerminal(repo, this.gitExecutable.path, command, name);
                setTimeout(() => resolve(null), 1000);
            }
        });
    }
    getBranches(repo, showRemoteBranches, hideRemotes) {
        let args = ['branch'];
        if (showRemoteBranches)
            args.push('-a');
        args.push('--no-color');
        const hideRemotePatterns = hideRemotes.map((remote) => 'remotes/' + remote + '/');
        const showRemoteHeads = config_1.getConfig().showRemoteHeads;
        return this.spawnGit(args, repo, (stdout) => {
            let branchData = { branches: [], head: null, error: null };
            let lines = stdout.split(EOL_REGEX);
            for (let i = 0; i < lines.length - 1; i++) {
                let name = lines[i].substring(2).split(' -> ')[0];
                if (INVALID_BRANCH_REGEXP.test(name) || hideRemotePatterns.some((pattern) => name.startsWith(pattern)) || (!showRemoteHeads && REMOTE_HEAD_BRANCH_REGEXP.test(name))) {
                    continue;
                }
                if (lines[i][0] === '*') {
                    branchData.head = name;
                    branchData.branches.unshift(name);
                }
                else {
                    branchData.branches.push(name);
                }
            }
            return branchData;
        });
    }
    getCommitDetailsBase(repo, commitHash) {
        return this.spawnGit(['-c', 'log.showSignature=false', 'show', '--quiet', commitHash, '--format=' + this.gitFormatCommitDetails], repo, (stdout) => {
            const commitInfo = stdout.split(GIT_LOG_SEPARATOR);
            return {
                hash: commitInfo[0],
                parents: commitInfo[1] !== '' ? commitInfo[1].split(' ') : [],
                author: commitInfo[2],
                authorEmail: commitInfo[3],
                authorDate: parseInt(commitInfo[4]),
                committer: commitInfo[5],
                committerEmail: commitInfo[6],
                committerDate: parseInt(commitInfo[7]),
                signature: ['G', 'U', 'X', 'Y', 'R', 'E', 'B'].includes(commitInfo[8])
                    ? {
                        key: commitInfo[10].trim(),
                        signer: commitInfo[9].trim(),
                        status: commitInfo[8]
                    }
                    : null,
                body: removeTrailingBlankLines(commitInfo.slice(11).join(GIT_LOG_SEPARATOR).split(EOL_REGEX)).join('\n'),
                fileChanges: []
            };
        });
    }
    getConfigList(repo, location) {
        const args = ['--no-pager', 'config', '--list', '-z', '--includes'];
        if (location) {
            args.push('--' + location);
        }
        return this.spawnGit(args, repo, (stdout) => {
            const configs = {}, keyValuePairs = stdout.split('\0');
            const numPairs = keyValuePairs.length - 1;
            let comps, key;
            for (let i = 0; i < numPairs; i++) {
                comps = keyValuePairs[i].split(EOL_REGEX);
                key = comps.shift();
                configs[key] = comps.join('\n');
            }
            return configs;
        }).catch((errorMessage) => {
            if (typeof errorMessage === 'string') {
                const message = errorMessage.toLowerCase();
                if (message.startsWith('fatal: unable to read config file') && message.endsWith('no such file or directory')) {
                    return {};
                }
            }
            else {
                errorMessage = 'An unexpected error occurred while spawning the Git child process.';
            }
            throw errorMessage;
        });
    }
    getDiffNameStatus(repo, fromHash, toHash, filter = 'AMDR') {
        return this.execDiff(repo, fromHash, toHash, '--name-status', filter).then((output) => {
            let records = [], i = 0;
            while (i < output.length && output[i] !== '') {
                let type = output[i][0];
                if (type === "A" || type === "D" || type === "M") {
                    let p = utils_1.getPathFromStr(output[i + 1]);
                    records.push({ type: type, oldFilePath: p, newFilePath: p });
                    i += 2;
                }
                else if (type === "R") {
                    records.push({ type: type, oldFilePath: utils_1.getPathFromStr(output[i + 1]), newFilePath: utils_1.getPathFromStr(output[i + 2]) });
                    i += 3;
                }
                else {
                    break;
                }
            }
            return records;
        });
    }
    getDiffNumStat(repo, fromHash, toHash, filter = 'AMDR') {
        return this.execDiff(repo, fromHash, toHash, '--numstat', filter).then((output) => {
            let records = [], i = 0;
            while (i < output.length && output[i] !== '') {
                let fields = output[i].split('\t');
                if (fields.length !== 3)
                    break;
                if (fields[2] !== '') {
                    records.push({ filePath: utils_1.getPathFromStr(fields[2]), additions: parseInt(fields[0]), deletions: parseInt(fields[1]) });
                    i += 1;
                }
                else {
                    records.push({ filePath: utils_1.getPathFromStr(output[i + 2]), additions: parseInt(fields[0]), deletions: parseInt(fields[1]) });
                    i += 3;
                }
            }
            return records;
        });
    }
    getLog(repo, branches, num, includeTags, includeRemotes, includeCommitsMentionedByReflogs, onlyFollowFirstParent, order, remotes, hideRemotes, stashes) {
        const args = ['-c', 'log.showSignature=false', 'log', '--max-count=' + num, '--format=' + this.gitFormatLog, '--' + order + '-order'];
        if (onlyFollowFirstParent) {
            args.push('--first-parent');
        }
        if (branches !== null) {
            for (let i = 0; i < branches.length; i++) {
                args.push(branches[i]);
            }
        }
        else {
            args.push('--branches');
            if (includeTags)
                args.push('--tags');
            if (includeCommitsMentionedByReflogs)
                args.push('--reflog');
            if (includeRemotes) {
                if (hideRemotes.length === 0) {
                    args.push('--remotes');
                }
                else {
                    remotes.filter((remote) => !hideRemotes.includes(remote)).forEach((remote) => {
                        args.push('--glob=refs/remotes/' + remote);
                    });
                }
            }
            const stashBaseHashes = stashes.map((stash) => stash.baseHash);
            stashBaseHashes.filter((hash, index) => stashBaseHashes.indexOf(hash) === index).forEach((hash) => args.push(hash));
            args.push('HEAD');
        }
        args.push('--');
        return this.spawnGit(args, repo, (stdout) => {
            let lines = stdout.split(EOL_REGEX);
            let commits = [];
            for (let i = 0; i < lines.length - 1; i++) {
                let line = lines[i].split(GIT_LOG_SEPARATOR);
                if (line.length !== 6)
                    break;
                commits.push({ hash: line[0], parents: line[1] !== '' ? line[1].split(' ') : [], author: line[2], email: line[3], date: parseInt(line[4]), message: line[5] });
            }
            return commits;
        });
    }
    getRefs(repo, showRemoteBranches, showRemoteHeads, hideRemotes) {
        let args = ['show-ref'];
        if (!showRemoteBranches)
            args.push('--heads', '--tags');
        args.push('-d', '--head');
        const hideRemotePatterns = hideRemotes.map((remote) => 'refs/remotes/' + remote + '/');
        return this.spawnGit(args, repo, (stdout) => {
            let refData = { head: null, heads: [], tags: [], remotes: [] };
            let lines = stdout.split(EOL_REGEX);
            for (let i = 0; i < lines.length - 1; i++) {
                let line = lines[i].split(' ');
                if (line.length < 2)
                    continue;
                let hash = line.shift();
                let ref = line.join(' ');
                if (ref.startsWith('refs/heads/')) {
                    refData.heads.push({ hash: hash, name: ref.substring(11) });
                }
                else if (ref.startsWith('refs/tags/')) {
                    let annotated = ref.endsWith('^{}');
                    refData.tags.push({ hash: hash, name: (annotated ? ref.substring(10, ref.length - 3) : ref.substring(10)), annotated: annotated });
                }
                else if (ref.startsWith('refs/remotes/')) {
                    if (!hideRemotePatterns.some((pattern) => ref.startsWith(pattern)) && (showRemoteHeads || !ref.endsWith('/HEAD'))) {
                        refData.remotes.push({ hash: hash, name: ref.substring(13) });
                    }
                }
                else if (ref === 'HEAD') {
                    refData.head = hash;
                }
            }
            return refData;
        });
    }
    getStashes(repo) {
        return this.spawnGit(['reflog', '--format=' + this.gitFormatStash, 'refs/stash', '--'], repo, (stdout) => {
            let lines = stdout.split(EOL_REGEX);
            let stashes = [];
            for (let i = 0; i < lines.length - 1; i++) {
                let line = lines[i].split(GIT_LOG_SEPARATOR);
                if (line.length !== 7 || line[1] === '')
                    continue;
                let parentHashes = line[1].split(' ');
                stashes.push({
                    hash: line[0],
                    baseHash: parentHashes[0],
                    untrackedFilesHash: parentHashes.length === 3 ? parentHashes[2] : null,
                    selector: line[2],
                    author: line[3],
                    email: line[4],
                    date: parseInt(line[5]),
                    message: line[6]
                });
            }
            return stashes;
        }).catch(() => []);
    }
    getRemotes(repo) {
        return this.spawnGit(['remote'], repo, (stdout) => {
            let lines = stdout.split(EOL_REGEX);
            lines.pop();
            return lines;
        });
    }
    getUncommittedChanges(repo) {
        return this.spawnGit(['status', '--untracked-files=' + (config_1.getConfig().showUntrackedFiles ? 'all' : 'no'), '--porcelain'], repo, (stdout) => {
            const numLines = stdout.split(EOL_REGEX).length;
            return numLines > 1 ? numLines - 1 : 0;
        });
    }
    getStatus(repo) {
        return this.spawnGit(['status', '-s', '--untracked-files=' + (config_1.getConfig().showUntrackedFiles ? 'all' : 'no'), '--porcelain', '-z'], repo, (stdout) => {
            let output = stdout.split('\0'), i = 0;
            let status = { deleted: [], untracked: [] };
            let path = '', c1 = '', c2 = '';
            while (i < output.length && output[i] !== '') {
                if (output[i].length < 4)
                    break;
                path = output[i].substring(3);
                c1 = output[i].substring(0, 1);
                c2 = output[i].substring(1, 2);
                if (c1 === 'D' || c2 === 'D')
                    status.deleted.push(path);
                else if (c1 === '?' || c2 === '?')
                    status.untracked.push(path);
                if (c1 === 'R' || c2 === 'R' || c1 === 'C' || c2 === 'C') {
                    i += 2;
                }
                else {
                    i += 1;
                }
            }
            return status;
        });
    }
    commitSquashIfStagedChangesExist(repo, obj, actionOn, squashMessageFormat, signCommits) {
        return this.areStagedChanges(repo).then((changes) => {
            if (changes) {
                const args = ['commit'];
                if (signCommits) {
                    args.push('-S');
                }
                if (squashMessageFormat === 0) {
                    args.push('-m', 'Merge ' + actionOn.toLowerCase() + ' \'' + obj + '\'');
                }
                else {
                    args.push('--no-edit');
                }
                return this.runGitCommand(args, repo);
            }
            else {
                return null;
            }
        });
    }
    execDiff(repo, fromHash, toHash, arg, filter) {
        let args;
        if (fromHash === toHash) {
            args = ['diff-tree', arg, '-r', '--root', '--find-renames', '--diff-filter=' + filter, '-z', fromHash];
        }
        else {
            args = ['diff', arg, '--find-renames', '--diff-filter=' + filter, '-z', fromHash];
            if (toHash !== '')
                args.push(toHash);
        }
        return this.spawnGit(args, repo, (stdout) => {
            let lines = stdout.split('\0');
            if (fromHash === toHash)
                lines.shift();
            return lines;
        });
    }
    runGitCommand(args, repo) {
        return this._spawnGit(args, repo, () => null).catch((errorMessage) => errorMessage);
    }
    spawnGit(args, repo, resolveValue) {
        return this._spawnGit(args, repo, (stdout) => resolveValue(stdout.toString()));
    }
    _spawnGit(args, repo, resolveValue) {
        return new Promise((resolve, reject) => {
            if (this.gitExecutable === null)
                return reject(utils_1.UNABLE_TO_FIND_GIT_MSG);
            utils_1.resolveSpawnOutput(cp.spawn(this.gitExecutable.path, args, {
                cwd: repo,
                env: Object.assign({}, process.env, this.askpassEnv)
            })).then((values) => {
                let status = values[0], stdout = values[1];
                if (status.code === 0) {
                    resolve(resolveValue(stdout));
                }
                else {
                    reject(getErrorMessage(status.error, stdout, values[2]));
                }
            });
            this.logger.logCmd('git', args);
        });
    }
}
exports.DataSource = DataSource;
function generateFileChanges(nameStatusRecords, numStatRecords, status) {
    let fileChanges = [], fileLookup = {}, i = 0;
    for (i = 0; i < nameStatusRecords.length; i++) {
        fileLookup[nameStatusRecords[i].newFilePath] = fileChanges.length;
        fileChanges.push({ oldFilePath: nameStatusRecords[i].oldFilePath, newFilePath: nameStatusRecords[i].newFilePath, type: nameStatusRecords[i].type, additions: null, deletions: null });
    }
    if (status !== null) {
        let filePath;
        for (i = 0; i < status.deleted.length; i++) {
            filePath = utils_1.getPathFromStr(status.deleted[i]);
            if (typeof fileLookup[filePath] === 'number') {
                fileChanges[fileLookup[filePath]].type = "D";
            }
            else {
                fileChanges.push({ oldFilePath: filePath, newFilePath: filePath, type: "D", additions: null, deletions: null });
            }
        }
        for (i = 0; i < status.untracked.length; i++) {
            filePath = utils_1.getPathFromStr(status.untracked[i]);
            fileChanges.push({ oldFilePath: filePath, newFilePath: filePath, type: "U", additions: null, deletions: null });
        }
    }
    for (i = 0; i < numStatRecords.length; i++) {
        if (typeof fileLookup[numStatRecords[i].filePath] === 'number') {
            fileChanges[fileLookup[numStatRecords[i].filePath]].additions = numStatRecords[i].additions;
            fileChanges[fileLookup[numStatRecords[i].filePath]].deletions = numStatRecords[i].deletions;
        }
    }
    return fileChanges;
}
function getConfigValue(configs, key) {
    return typeof configs[key] !== 'undefined' ? configs[key] : null;
}
function getErrorMessage(error, stdoutBuffer, stderr) {
    let stdout = stdoutBuffer.toString(), lines;
    if (stdout !== '' || stderr !== '') {
        lines = (stderr + stdout).split(EOL_REGEX);
        lines.pop();
    }
    else if (error) {
        lines = error.message.split(EOL_REGEX);
    }
    else {
        lines = [];
    }
    return lines.join('\n');
}
function removeTrailingBlankLines(lines) {
    while (lines.length > 0 && lines[lines.length - 1] === '') {
        lines.pop();
    }
    return lines;
}
function unique(items) {
    const uniqueItems = {};
    items.forEach((item) => uniqueItems[item] = true);
    return Object.keys(uniqueItems);
}
//# sourceMappingURL=dataSource.js.map