"use strict";
function requireWithFallback(electronModule, nodeModule) { try { return require(electronModule); } catch (err) {} return require(nodeModule); }
Object.defineProperty(exports, "__esModule", { value: true });
exports.ExtensionState = exports.DEFAULT_REPO_STATE = void 0;
const fs = requireWithFallback("original-fs", "fs");
const config_1 = require("./config");
const utils_1 = require("./utils");
const disposable_1 = require("./utils/disposable");
const AVATAR_STORAGE_FOLDER = '/avatars';
const AVATAR_CACHE = 'avatarCache';
const CODE_REVIEWS = 'codeReviews';
const GLOBAL_VIEW_STATE = 'globalViewState';
const IGNORED_REPOS = 'ignoredRepos';
const LAST_ACTIVE_REPO = 'lastActiveRepo';
const LAST_KNOWN_GIT_PATH = 'lastKnownGitPath';
const REPO_STATES = 'repoStates';
const WORKSPACE_VIEW_STATE = 'workspaceViewState';
exports.DEFAULT_REPO_STATE = {
    cdvDivider: 0.5,
    cdvHeight: 250,
    columnWidths: null,
    commitOrdering: "default",
    fileViewType: 0,
    hideRemotes: [],
    includeCommitsMentionedByReflogs: 0,
    issueLinkingConfig: null,
    lastImportAt: 0,
    name: null,
    onlyFollowFirstParent: 0,
    onRepoLoadShowCheckedOutBranch: 0,
    onRepoLoadShowSpecificBranches: null,
    pullRequestConfig: null,
    showRemoteBranches: true,
    showRemoteBranchesV2: 0,
    showStashes: 0,
    showTags: 0,
    workspaceFolderIndex: null
};
const DEFAULT_GIT_GRAPH_VIEW_GLOBAL_STATE = {
    alwaysAcceptCheckoutCommit: false,
    issueLinkingConfig: null
};
const DEFAULT_GIT_GRAPH_VIEW_WORKSPACE_STATE = {
    findIsCaseSensitive: false,
    findIsRegex: false,
    findOpenCommitDetailsView: false
};
class ExtensionState extends disposable_1.Disposable {
    constructor(context, onDidChangeGitExecutable) {
        super();
        this.avatarStorageAvailable = false;
        this.globalState = context.globalState;
        this.workspaceState = context.workspaceState;
        this.globalStoragePath = utils_1.getPathFromStr(context.globalStoragePath);
        fs.stat(this.globalStoragePath + AVATAR_STORAGE_FOLDER, (err) => {
            if (!err) {
                this.avatarStorageAvailable = true;
            }
            else {
                fs.mkdir(this.globalStoragePath, () => {
                    fs.mkdir(this.globalStoragePath + AVATAR_STORAGE_FOLDER, (err) => {
                        if (!err || err.code === 'EEXIST') {
                            this.avatarStorageAvailable = true;
                        }
                    });
                });
            }
        });
        this.registerDisposable(onDidChangeGitExecutable((gitExecutable) => {
            this.setLastKnownGitPath(gitExecutable.path);
        }));
    }
    getRepos() {
        const repoSet = this.workspaceState.get(REPO_STATES, {});
        const outputSet = {};
        let showRemoteBranchesDefaultValue = null;
        Object.keys(repoSet).forEach((repo) => {
            outputSet[repo] = Object.assign({}, exports.DEFAULT_REPO_STATE, repoSet[repo]);
            if (typeof repoSet[repo].showRemoteBranchesV2 === 'undefined' && typeof repoSet[repo].showRemoteBranches !== 'undefined') {
                if (showRemoteBranchesDefaultValue === null) {
                    showRemoteBranchesDefaultValue = config_1.getConfig().showRemoteBranches;
                }
                if (repoSet[repo].showRemoteBranches !== showRemoteBranchesDefaultValue) {
                    outputSet[repo].showRemoteBranchesV2 = repoSet[repo].showRemoteBranches ? 1 : 2;
                }
            }
        });
        return outputSet;
    }
    saveRepos(gitRepoSet) {
        this.updateWorkspaceState(REPO_STATES, gitRepoSet);
    }
    transferRepo(oldRepo, newRepo) {
        if (this.getLastActiveRepo() === oldRepo) {
            this.setLastActiveRepo(newRepo);
        }
        let reviews = this.getCodeReviews();
        if (typeof reviews[oldRepo] !== 'undefined') {
            reviews[newRepo] = reviews[oldRepo];
            delete reviews[oldRepo];
            this.setCodeReviews(reviews);
        }
    }
    getGlobalViewState() {
        const globalViewState = this.globalState.get(GLOBAL_VIEW_STATE, DEFAULT_GIT_GRAPH_VIEW_GLOBAL_STATE);
        return Object.assign({}, DEFAULT_GIT_GRAPH_VIEW_GLOBAL_STATE, globalViewState);
    }
    setGlobalViewState(state) {
        return this.updateGlobalState(GLOBAL_VIEW_STATE, state);
    }
    getWorkspaceViewState() {
        const workspaceViewState = this.workspaceState.get(WORKSPACE_VIEW_STATE, DEFAULT_GIT_GRAPH_VIEW_WORKSPACE_STATE);
        return Object.assign({}, DEFAULT_GIT_GRAPH_VIEW_WORKSPACE_STATE, workspaceViewState);
    }
    setWorkspaceViewState(state) {
        return this.updateWorkspaceState(WORKSPACE_VIEW_STATE, state);
    }
    getIgnoredRepos() {
        return this.workspaceState.get(IGNORED_REPOS, []);
    }
    setIgnoredRepos(ignoredRepos) {
        return this.updateWorkspaceState(IGNORED_REPOS, ignoredRepos);
    }
    getLastActiveRepo() {
        return this.workspaceState.get(LAST_ACTIVE_REPO, null);
    }
    setLastActiveRepo(repo) {
        this.updateWorkspaceState(LAST_ACTIVE_REPO, repo);
    }
    getLastKnownGitPath() {
        return this.globalState.get(LAST_KNOWN_GIT_PATH, null);
    }
    setLastKnownGitPath(path) {
        this.updateGlobalState(LAST_KNOWN_GIT_PATH, path);
    }
    isAvatarStorageAvailable() {
        return this.avatarStorageAvailable;
    }
    getAvatarStoragePath() {
        return this.globalStoragePath + AVATAR_STORAGE_FOLDER;
    }
    getAvatarCache() {
        return this.globalState.get(AVATAR_CACHE, {});
    }
    saveAvatar(email, avatar) {
        let avatars = this.getAvatarCache();
        avatars[email] = avatar;
        this.updateGlobalState(AVATAR_CACHE, avatars);
    }
    removeAvatarFromCache(email) {
        let avatars = this.getAvatarCache();
        delete avatars[email];
        this.updateGlobalState(AVATAR_CACHE, avatars);
    }
    clearAvatarCache() {
        this.updateGlobalState(AVATAR_CACHE, {});
        fs.readdir(this.globalStoragePath + AVATAR_STORAGE_FOLDER, (err, files) => {
            if (err)
                return;
            for (let i = 0; i < files.length; i++) {
                fs.unlink(this.globalStoragePath + AVATAR_STORAGE_FOLDER + '/' + files[i], () => { });
            }
        });
    }
    startCodeReview(repo, id, files, lastViewedFile) {
        let reviews = this.getCodeReviews();
        if (typeof reviews[repo] === 'undefined')
            reviews[repo] = {};
        reviews[repo][id] = { lastActive: (new Date()).getTime(), lastViewedFile: lastViewedFile, remainingFiles: files };
        return this.setCodeReviews(reviews).then((err) => ({
            codeReview: Object.assign({ id: id }, reviews[repo][id]),
            error: err
        }));
    }
    endCodeReview(repo, id) {
        let reviews = this.getCodeReviews();
        removeCodeReview(reviews, repo, id);
        return this.setCodeReviews(reviews);
    }
    getCodeReview(repo, id) {
        let reviews = this.getCodeReviews();
        if (typeof reviews[repo] !== 'undefined' && typeof reviews[repo][id] !== 'undefined') {
            reviews[repo][id].lastActive = (new Date()).getTime();
            this.setCodeReviews(reviews);
            return Object.assign({ id: id }, reviews[repo][id]);
        }
        else {
            return null;
        }
    }
    updateCodeReview(repo, id, remainingFiles, lastViewedFile) {
        const reviews = this.getCodeReviews();
        if (typeof reviews[repo] === 'undefined' || typeof reviews[repo][id] === 'undefined') {
            return Promise.resolve('The Code Review could not be found.');
        }
        if (remainingFiles.length > 0) {
            reviews[repo][id].remainingFiles = remainingFiles;
            reviews[repo][id].lastActive = (new Date()).getTime();
            if (lastViewedFile !== null) {
                reviews[repo][id].lastViewedFile = lastViewedFile;
            }
        }
        else {
            removeCodeReview(reviews, repo, id);
        }
        return this.setCodeReviews(reviews);
    }
    expireOldCodeReviews() {
        let reviews = this.getCodeReviews(), change = false, expireReviewsBefore = (new Date()).getTime() - 7776000000;
        Object.keys(reviews).forEach((repo) => {
            Object.keys(reviews[repo]).forEach((id) => {
                if (reviews[repo][id].lastActive < expireReviewsBefore) {
                    delete reviews[repo][id];
                    change = true;
                }
            });
            removeCodeReviewRepoIfEmpty(reviews, repo);
        });
        if (change)
            this.setCodeReviews(reviews);
    }
    endAllWorkspaceCodeReviews() {
        this.setCodeReviews({});
    }
    getCodeReviews() {
        return this.workspaceState.get(CODE_REVIEWS, {});
    }
    setCodeReviews(reviews) {
        return this.updateWorkspaceState(CODE_REVIEWS, reviews);
    }
    updateGlobalState(key, value) {
        return this.globalState.update(key, value).then(() => null, () => 'Visual Studio Code was unable to save the Git Graph Global State Memento.');
    }
    updateWorkspaceState(key, value) {
        return this.workspaceState.update(key, value).then(() => null, () => 'Visual Studio Code was unable to save the Git Graph Workspace State Memento.');
    }
}
exports.ExtensionState = ExtensionState;
function removeCodeReview(reviews, repo, id) {
    if (typeof reviews[repo] !== 'undefined' && typeof reviews[repo][id] !== 'undefined') {
        delete reviews[repo][id];
        removeCodeReviewRepoIfEmpty(reviews, repo);
    }
}
function removeCodeReviewRepoIfEmpty(reviews, repo) {
    if (typeof reviews[repo] !== 'undefined' && Object.keys(reviews[repo]).length === 0) {
        delete reviews[repo];
    }
}
//# sourceMappingURL=extensionState.js.map