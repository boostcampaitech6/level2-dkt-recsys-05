"use strict";
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
exports.standardiseCspSource = exports.GitGraphView = void 0;
const path = require("path");
const vscode = require("vscode");
const config_1 = require("./config");
const dataSource_1 = require("./dataSource");
const repoFileWatcher_1 = require("./repoFileWatcher");
const utils_1 = require("./utils");
const disposable_1 = require("./utils/disposable");
class GitGraphView extends disposable_1.Disposable {
    constructor(extensionPath, dataSource, extensionState, avatarManager, repoManager, logger, loadViewTo, column) {
        super();
        this.isGraphViewLoaded = false;
        this.isPanelVisible = true;
        this.currentRepo = null;
        this.loadViewTo = null;
        this.loadRepoInfoRefreshId = 0;
        this.loadCommitsRefreshId = 0;
        this.extensionPath = extensionPath;
        this.avatarManager = avatarManager;
        this.dataSource = dataSource;
        this.extensionState = extensionState;
        this.repoManager = repoManager;
        this.logger = logger;
        this.loadViewTo = loadViewTo;
        const config = config_1.getConfig();
        this.panel = vscode.window.createWebviewPanel('git-graph', 'Git Graph', column || vscode.ViewColumn.One, {
            enableScripts: true,
            localResourceRoots: [vscode.Uri.file(path.join(extensionPath, 'media'))],
            retainContextWhenHidden: config.retainContextWhenHidden
        });
        this.panel.iconPath = config.tabIconColourTheme === 0
            ? this.getResourcesUri('webview-icon.svg')
            : {
                light: this.getResourcesUri('webview-icon-light.svg'),
                dark: this.getResourcesUri('webview-icon-dark.svg')
            };
        this.registerDisposables(disposable_1.toDisposable(() => {
            GitGraphView.currentPanel = undefined;
            this.repoFileWatcher.stop();
        }), this.panel.onDidDispose(() => this.dispose()), this.panel.onDidChangeViewState(() => {
            if (this.panel.visible !== this.isPanelVisible) {
                if (this.panel.visible) {
                    this.update();
                }
                else {
                    this.currentRepo = null;
                    this.repoFileWatcher.stop();
                }
                this.isPanelVisible = this.panel.visible;
            }
        }), repoManager.onDidChangeRepos((event) => {
            if (!this.panel.visible)
                return;
            const loadViewTo = event.loadRepo !== null ? { repo: event.loadRepo } : null;
            if ((event.numRepos === 0 && this.isGraphViewLoaded) || (event.numRepos > 0 && !this.isGraphViewLoaded)) {
                this.loadViewTo = loadViewTo;
                this.update();
            }
            else {
                this.respondLoadRepos(event.repos, loadViewTo);
            }
        }), avatarManager.onAvatar((event) => {
            this.sendMessage({
                command: 'fetchAvatar',
                email: event.email,
                image: event.image
            });
        }), this.panel.webview.onDidReceiveMessage((msg) => this.respondToMessage(msg)), this.panel);
        this.repoFileWatcher = new repoFileWatcher_1.RepoFileWatcher(logger, () => {
            if (this.panel.visible) {
                this.sendMessage({ command: 'refresh' });
            }
        });
        this.update();
        this.logger.log('Created Git Graph View' + (loadViewTo !== null ? ' (active repo: ' + loadViewTo.repo + ')' : ''));
    }
    static createOrShow(extensionPath, dataSource, extensionState, avatarManager, repoManager, logger, loadViewTo) {
        const column = vscode.window.activeTextEditor ? vscode.window.activeTextEditor.viewColumn : undefined;
        if (GitGraphView.currentPanel) {
            if (GitGraphView.currentPanel.isPanelVisible) {
                if (loadViewTo !== null) {
                    GitGraphView.currentPanel.respondLoadRepos(repoManager.getRepos(), loadViewTo);
                }
            }
            else {
                GitGraphView.currentPanel.loadViewTo = loadViewTo;
            }
            GitGraphView.currentPanel.panel.reveal(column);
        }
        else {
            GitGraphView.currentPanel = new GitGraphView(extensionPath, dataSource, extensionState, avatarManager, repoManager, logger, loadViewTo, column);
        }
    }
    respondToMessage(msg) {
        return __awaiter(this, void 0, void 0, function* () {
            this.repoFileWatcher.mute();
            let errorInfos;
            switch (msg.command) {
                case 'addRemote':
                    this.sendMessage({
                        command: 'addRemote',
                        error: yield this.dataSource.addRemote(msg.repo, msg.name, msg.url, msg.pushUrl, msg.fetch)
                    });
                    break;
                case 'addTag':
                    errorInfos = [yield this.dataSource.addTag(msg.repo, msg.tagName, msg.commitHash, msg.type, msg.message, msg.force)];
                    if (errorInfos[0] === null && msg.pushToRemote !== null) {
                        errorInfos.push(yield this.dataSource.pushTag(msg.repo, msg.tagName, msg.pushToRemote));
                    }
                    this.sendMessage({ command: 'addTag', errors: errorInfos });
                    break;
                case 'applyStash':
                    this.sendMessage({
                        command: 'applyStash',
                        error: yield this.dataSource.applyStash(msg.repo, msg.selector, msg.reinstateIndex)
                    });
                    break;
                case 'branchFromStash':
                    this.sendMessage({
                        command: 'branchFromStash',
                        error: yield this.dataSource.branchFromStash(msg.repo, msg.selector, msg.branchName)
                    });
                    break;
                case 'checkoutBranch':
                    errorInfos = [yield this.dataSource.checkoutBranch(msg.repo, msg.branchName, msg.remoteBranch)];
                    if (errorInfos[0] === null && msg.pullAfterwards !== null) {
                        errorInfos.push(yield this.dataSource.pullBranch(msg.repo, msg.pullAfterwards.branchName, msg.pullAfterwards.remote, msg.pullAfterwards.createNewCommit, msg.pullAfterwards.squash));
                    }
                    this.sendMessage({
                        command: 'checkoutBranch',
                        pullAfterwards: msg.pullAfterwards,
                        errors: errorInfos
                    });
                    break;
                case 'checkoutCommit':
                    this.sendMessage({
                        command: 'checkoutCommit',
                        error: yield this.dataSource.checkoutCommit(msg.repo, msg.commitHash)
                    });
                    break;
                case 'cherrypickCommit':
                    errorInfos = [yield this.dataSource.cherrypickCommit(msg.repo, msg.commitHash, msg.parentIndex, msg.recordOrigin, msg.noCommit)];
                    if (errorInfos[0] === null && msg.noCommit) {
                        errorInfos.push(yield utils_1.viewScm());
                    }
                    this.sendMessage({ command: 'cherrypickCommit', errors: errorInfos });
                    break;
                case 'cleanUntrackedFiles':
                    this.sendMessage({
                        command: 'cleanUntrackedFiles',
                        error: yield this.dataSource.cleanUntrackedFiles(msg.repo, msg.directories)
                    });
                    break;
                case 'commitDetails':
                    let data = yield Promise.all([
                        msg.commitHash === utils_1.UNCOMMITTED
                            ? this.dataSource.getUncommittedDetails(msg.repo)
                            : msg.stash === null
                                ? this.dataSource.getCommitDetails(msg.repo, msg.commitHash, msg.hasParents)
                                : this.dataSource.getStashDetails(msg.repo, msg.commitHash, msg.stash),
                        msg.avatarEmail !== null ? this.avatarManager.getAvatarImage(msg.avatarEmail) : Promise.resolve(null)
                    ]);
                    this.sendMessage(Object.assign(Object.assign({ command: 'commitDetails' }, data[0]), { avatar: data[1], codeReview: msg.commitHash !== utils_1.UNCOMMITTED ? this.extensionState.getCodeReview(msg.repo, msg.commitHash) : null, refresh: msg.refresh }));
                    break;
                case 'compareCommits':
                    this.sendMessage(Object.assign(Object.assign({ command: 'compareCommits', commitHash: msg.commitHash, compareWithHash: msg.compareWithHash }, yield this.dataSource.getCommitComparison(msg.repo, msg.fromHash, msg.toHash)), { codeReview: msg.toHash !== utils_1.UNCOMMITTED ? this.extensionState.getCodeReview(msg.repo, msg.fromHash + '-' + msg.toHash) : null, refresh: msg.refresh }));
                    break;
                case 'copyFilePath':
                    this.sendMessage({
                        command: 'copyFilePath',
                        error: yield utils_1.copyFilePathToClipboard(msg.repo, msg.filePath, msg.absolute)
                    });
                    break;
                case 'copyToClipboard':
                    this.sendMessage({
                        command: 'copyToClipboard',
                        type: msg.type,
                        error: yield utils_1.copyToClipboard(msg.data)
                    });
                    break;
                case 'createArchive':
                    this.sendMessage({
                        command: 'createArchive',
                        error: yield utils_1.archive(msg.repo, msg.ref, this.dataSource)
                    });
                    break;
                case 'createBranch':
                    this.sendMessage({
                        command: 'createBranch',
                        errors: yield this.dataSource.createBranch(msg.repo, msg.branchName, msg.commitHash, msg.checkout, msg.force)
                    });
                    break;
                case 'createPullRequest':
                    errorInfos = [msg.push ? yield this.dataSource.pushBranch(msg.repo, msg.sourceBranch, msg.sourceRemote, true, "") : null];
                    if (errorInfos[0] === null) {
                        errorInfos.push(yield utils_1.createPullRequest(msg.config, msg.sourceOwner, msg.sourceRepo, msg.sourceBranch));
                    }
                    this.sendMessage({
                        command: 'createPullRequest',
                        push: msg.push,
                        errors: errorInfos
                    });
                    break;
                case 'deleteBranch':
                    errorInfos = [yield this.dataSource.deleteBranch(msg.repo, msg.branchName, msg.forceDelete)];
                    if (errorInfos[0] === null) {
                        for (let i = 0; i < msg.deleteOnRemotes.length; i++) {
                            errorInfos.push(yield this.dataSource.deleteRemoteBranch(msg.repo, msg.branchName, msg.deleteOnRemotes[i]));
                        }
                    }
                    this.sendMessage({
                        command: 'deleteBranch',
                        repo: msg.repo,
                        branchName: msg.branchName,
                        deleteOnRemotes: msg.deleteOnRemotes,
                        errors: errorInfos
                    });
                    break;
                case 'deleteRemote':
                    this.sendMessage({
                        command: 'deleteRemote',
                        error: yield this.dataSource.deleteRemote(msg.repo, msg.name)
                    });
                    break;
                case 'deleteRemoteBranch':
                    this.sendMessage({
                        command: 'deleteRemoteBranch',
                        error: yield this.dataSource.deleteRemoteBranch(msg.repo, msg.branchName, msg.remote)
                    });
                    break;
                case 'deleteTag':
                    this.sendMessage({
                        command: 'deleteTag',
                        error: yield this.dataSource.deleteTag(msg.repo, msg.tagName, msg.deleteOnRemote)
                    });
                    break;
                case 'deleteUserDetails':
                    errorInfos = [];
                    if (msg.name) {
                        errorInfos.push(yield this.dataSource.unsetConfigValue(msg.repo, dataSource_1.GIT_CONFIG.USER.NAME, msg.location));
                    }
                    if (msg.email) {
                        errorInfos.push(yield this.dataSource.unsetConfigValue(msg.repo, dataSource_1.GIT_CONFIG.USER.EMAIL, msg.location));
                    }
                    this.sendMessage({
                        command: 'deleteUserDetails',
                        errors: errorInfos
                    });
                    break;
                case 'dropCommit':
                    this.sendMessage({
                        command: 'dropCommit',
                        error: yield this.dataSource.dropCommit(msg.repo, msg.commitHash)
                    });
                    break;
                case 'dropStash':
                    this.sendMessage({
                        command: 'dropStash',
                        error: yield this.dataSource.dropStash(msg.repo, msg.selector)
                    });
                    break;
                case 'editRemote':
                    this.sendMessage({
                        command: 'editRemote',
                        error: yield this.dataSource.editRemote(msg.repo, msg.nameOld, msg.nameNew, msg.urlOld, msg.urlNew, msg.pushUrlOld, msg.pushUrlNew)
                    });
                    break;
                case 'editUserDetails':
                    errorInfos = [
                        yield this.dataSource.setConfigValue(msg.repo, dataSource_1.GIT_CONFIG.USER.NAME, msg.name, msg.location),
                        yield this.dataSource.setConfigValue(msg.repo, dataSource_1.GIT_CONFIG.USER.EMAIL, msg.email, msg.location)
                    ];
                    if (errorInfos[0] === null && errorInfos[1] === null) {
                        if (msg.deleteLocalName) {
                            errorInfos.push(yield this.dataSource.unsetConfigValue(msg.repo, dataSource_1.GIT_CONFIG.USER.NAME, "local"));
                        }
                        if (msg.deleteLocalEmail) {
                            errorInfos.push(yield this.dataSource.unsetConfigValue(msg.repo, dataSource_1.GIT_CONFIG.USER.EMAIL, "local"));
                        }
                    }
                    this.sendMessage({
                        command: 'editUserDetails',
                        errors: errorInfos
                    });
                    break;
                case 'fetch':
                    this.sendMessage({
                        command: 'fetch',
                        error: yield this.dataSource.fetch(msg.repo, msg.name, msg.prune, msg.pruneTags)
                    });
                    break;
                case 'fetchAvatar':
                    this.avatarManager.fetchAvatarImage(msg.email, msg.repo, msg.remote, msg.commits);
                    break;
                case 'fetchIntoLocalBranch':
                    this.sendMessage({
                        command: 'fetchIntoLocalBranch',
                        error: yield this.dataSource.fetchIntoLocalBranch(msg.repo, msg.remote, msg.remoteBranch, msg.localBranch, msg.force)
                    });
                    break;
                case 'endCodeReview':
                    this.extensionState.endCodeReview(msg.repo, msg.id);
                    break;
                case 'exportRepoConfig':
                    this.sendMessage({
                        command: 'exportRepoConfig',
                        error: yield this.repoManager.exportRepoConfig(msg.repo)
                    });
                    break;
                case 'loadCommits':
                    this.loadCommitsRefreshId = msg.refreshId;
                    this.sendMessage(Object.assign({ command: 'loadCommits', refreshId: msg.refreshId, onlyFollowFirstParent: msg.onlyFollowFirstParent }, yield this.dataSource.getCommits(msg.repo, msg.branches, msg.maxCommits, msg.showTags, msg.showRemoteBranches, msg.includeCommitsMentionedByReflogs, msg.onlyFollowFirstParent, msg.commitOrdering, msg.remotes, msg.hideRemotes, msg.stashes)));
                    break;
                case 'loadConfig':
                    this.sendMessage(Object.assign({ command: 'loadConfig', repo: msg.repo }, yield this.dataSource.getConfig(msg.repo, msg.remotes)));
                    break;
                case 'loadRepoInfo':
                    this.loadRepoInfoRefreshId = msg.refreshId;
                    let repoInfo = yield this.dataSource.getRepoInfo(msg.repo, msg.showRemoteBranches, msg.showStashes, msg.hideRemotes), isRepo = true;
                    if (repoInfo.error) {
                        isRepo = (yield this.dataSource.repoRoot(msg.repo)) !== null;
                        if (!isRepo)
                            repoInfo.error = null;
                    }
                    this.sendMessage(Object.assign(Object.assign({ command: 'loadRepoInfo', refreshId: msg.refreshId }, repoInfo), { isRepo: isRepo }));
                    if (msg.repo !== this.currentRepo) {
                        this.currentRepo = msg.repo;
                        this.extensionState.setLastActiveRepo(msg.repo);
                        this.repoFileWatcher.start(msg.repo);
                    }
                    break;
                case 'loadRepos':
                    if (!msg.check || !(yield this.repoManager.checkReposExist())) {
                        this.respondLoadRepos(this.repoManager.getRepos(), null);
                    }
                    break;
                case 'merge':
                    this.sendMessage({
                        command: 'merge', actionOn: msg.actionOn,
                        error: yield this.dataSource.merge(msg.repo, msg.obj, msg.actionOn, msg.createNewCommit, msg.squash, msg.noCommit)
                    });
                    break;
                case 'openExtensionSettings':
                    this.sendMessage({
                        command: 'openExtensionSettings',
                        error: yield utils_1.openExtensionSettings()
                    });
                    break;
                case 'openExternalDirDiff':
                    this.sendMessage({
                        command: 'openExternalDirDiff',
                        error: yield this.dataSource.openExternalDirDiff(msg.repo, msg.fromHash, msg.toHash, msg.isGui)
                    });
                    break;
                case 'openExternalUrl':
                    this.sendMessage({
                        command: 'openExternalUrl',
                        error: yield utils_1.openExternalUrl(msg.url)
                    });
                    break;
                case 'openFile':
                    this.sendMessage({
                        command: 'openFile',
                        error: yield utils_1.openFile(msg.repo, msg.filePath, msg.hash, this.dataSource)
                    });
                    break;
                case 'openTerminal':
                    this.sendMessage({
                        command: 'openTerminal',
                        error: yield this.dataSource.openGitTerminal(msg.repo, null, msg.name)
                    });
                    break;
                case 'popStash':
                    this.sendMessage({
                        command: 'popStash',
                        error: yield this.dataSource.popStash(msg.repo, msg.selector, msg.reinstateIndex)
                    });
                    break;
                case 'pruneRemote':
                    this.sendMessage({
                        command: 'pruneRemote',
                        error: yield this.dataSource.pruneRemote(msg.repo, msg.name)
                    });
                    break;
                case 'pullBranch':
                    this.sendMessage({
                        command: 'pullBranch',
                        error: yield this.dataSource.pullBranch(msg.repo, msg.branchName, msg.remote, msg.createNewCommit, msg.squash)
                    });
                    break;
                case 'pushBranch':
                    this.sendMessage({
                        command: 'pushBranch',
                        willUpdateBranchConfig: msg.willUpdateBranchConfig,
                        errors: yield this.dataSource.pushBranchToMultipleRemotes(msg.repo, msg.branchName, msg.remotes, msg.setUpstream, msg.mode)
                    });
                    break;
                case 'pushStash':
                    this.sendMessage({
                        command: 'pushStash',
                        error: yield this.dataSource.pushStash(msg.repo, msg.message, msg.includeUntracked)
                    });
                    break;
                case 'pushTag':
                    this.sendMessage({
                        command: 'pushTag',
                        errors: yield this.dataSource.pushTagToMultipleRemotes(msg.repo, msg.tagName, msg.remotes)
                    });
                    break;
                case 'rebase':
                    this.sendMessage({
                        command: 'rebase', actionOn: msg.actionOn, interactive: msg.interactive,
                        error: yield this.dataSource.rebase(msg.repo, msg.obj, msg.actionOn, msg.ignoreDate, msg.interactive)
                    });
                    break;
                case 'renameBranch':
                    this.sendMessage({
                        command: 'renameBranch',
                        error: yield this.dataSource.renameBranch(msg.repo, msg.oldName, msg.newName)
                    });
                    break;
                case 'rescanForRepos':
                    if (!(yield this.repoManager.searchWorkspaceForRepos())) {
                        utils_1.showErrorMessage('No Git repositories were found in the current workspace.');
                    }
                    break;
                case 'resetToCommit':
                    this.sendMessage({
                        command: 'resetToCommit',
                        error: yield this.dataSource.resetToCommit(msg.repo, msg.commit, msg.resetMode)
                    });
                    break;
                case 'revertCommit':
                    this.sendMessage({
                        command: 'revertCommit',
                        error: yield this.dataSource.revertCommit(msg.repo, msg.commitHash, msg.parentIndex)
                    });
                    break;
                case 'setGlobalViewState':
                    this.sendMessage({
                        command: 'setGlobalViewState',
                        error: yield this.extensionState.setGlobalViewState(msg.state)
                    });
                    break;
                case 'setRepoState':
                    this.repoManager.setRepoState(msg.repo, msg.state);
                    break;
                case 'setWorkspaceViewState':
                    this.sendMessage({
                        command: 'setWorkspaceViewState',
                        error: yield this.extensionState.setWorkspaceViewState(msg.state)
                    });
                    break;
                case 'showErrorMessage':
                    utils_1.showErrorMessage(msg.message);
                    break;
                case 'startCodeReview':
                    this.sendMessage(Object.assign({ command: 'startCodeReview', commitHash: msg.commitHash, compareWithHash: msg.compareWithHash }, yield this.extensionState.startCodeReview(msg.repo, msg.id, msg.files, msg.lastViewedFile)));
                    break;
                case 'tagDetails':
                    this.sendMessage(Object.assign({ command: 'tagDetails', tagName: msg.tagName, commitHash: msg.commitHash }, yield this.dataSource.getTagDetails(msg.repo, msg.tagName)));
                    break;
                case 'updateCodeReview':
                    this.sendMessage({
                        command: 'updateCodeReview',
                        error: yield this.extensionState.updateCodeReview(msg.repo, msg.id, msg.remainingFiles, msg.lastViewedFile)
                    });
                    break;
                case 'viewDiff':
                    this.sendMessage({
                        command: 'viewDiff',
                        error: yield utils_1.viewDiff(msg.repo, msg.fromHash, msg.toHash, msg.oldFilePath, msg.newFilePath, msg.type)
                    });
                    break;
                case 'viewDiffWithWorkingFile':
                    this.sendMessage({
                        command: 'viewDiffWithWorkingFile',
                        error: yield utils_1.viewDiffWithWorkingFile(msg.repo, msg.hash, msg.filePath, this.dataSource)
                    });
                    break;
                case 'viewFileAtRevision':
                    this.sendMessage({
                        command: 'viewFileAtRevision',
                        error: yield utils_1.viewFileAtRevision(msg.repo, msg.hash, msg.filePath)
                    });
                    break;
                case 'viewScm':
                    this.sendMessage({
                        command: 'viewScm',
                        error: yield utils_1.viewScm()
                    });
                    break;
            }
            this.repoFileWatcher.unmute();
        });
    }
    sendMessage(msg) {
        this.panel.webview.postMessage(msg);
    }
    update() {
        this.panel.webview.html = this.getHtmlForWebview();
    }
    getHtmlForWebview() {
        const config = config_1.getConfig(), nonce = utils_1.getNonce();
        const initialState = {
            config: {
                commitDetailsView: config.commitDetailsView,
                commitOrdering: config.commitOrder,
                contextMenuActionsVisibility: config.contextMenuActionsVisibility,
                customBranchGlobPatterns: config.customBranchGlobPatterns,
                customEmojiShortcodeMappings: config.customEmojiShortcodeMappings,
                customPullRequestProviders: config.customPullRequestProviders,
                dateFormat: config.dateFormat,
                defaultColumnVisibility: config.defaultColumnVisibility,
                dialogDefaults: config.dialogDefaults,
                enhancedAccessibility: config.enhancedAccessibility,
                fetchAndPrune: config.fetchAndPrune,
                fetchAndPruneTags: config.fetchAndPruneTags,
                fetchAvatars: config.fetchAvatars && this.extensionState.isAvatarStorageAvailable(),
                graph: config.graph,
                includeCommitsMentionedByReflogs: config.includeCommitsMentionedByReflogs,
                initialLoadCommits: config.initialLoadCommits,
                keybindings: config.keybindings,
                loadMoreCommits: config.loadMoreCommits,
                loadMoreCommitsAutomatically: config.loadMoreCommitsAutomatically,
                markdown: config.markdown,
                mute: config.muteCommits,
                onlyFollowFirstParent: config.onlyFollowFirstParent,
                onRepoLoad: config.onRepoLoad,
                referenceLabels: config.referenceLabels,
                repoDropdownOrder: config.repoDropdownOrder,
                showRemoteBranches: config.showRemoteBranches,
                showStashes: config.showStashes,
                showTags: config.showTags
            },
            lastActiveRepo: this.extensionState.getLastActiveRepo(),
            loadViewTo: this.loadViewTo,
            repos: this.repoManager.getRepos(),
            loadRepoInfoRefreshId: this.loadRepoInfoRefreshId,
            loadCommitsRefreshId: this.loadCommitsRefreshId
        };
        const globalState = this.extensionState.getGlobalViewState();
        const workspaceState = this.extensionState.getWorkspaceViewState();
        let body, numRepos = Object.keys(initialState.repos).length, colorVars = '', colorParams = '';
        for (let i = 0; i < initialState.config.graph.colours.length; i++) {
            colorVars += '--git-graph-color' + i + ':' + initialState.config.graph.colours[i] + '; ';
            colorParams += '[data-color="' + i + '"]{--git-graph-color:var(--git-graph-color' + i + ');} ';
        }
        if (this.dataSource.isGitExecutableUnknown()) {
            body = `<body class="unableToLoad">
			<h2>Unable to load Git Graph</h2>
			<p class="unableToLoadMessage">${utils_1.UNABLE_TO_FIND_GIT_MSG}</p>
			</body>`;
        }
        else if (numRepos > 0) {
            body = `<body>
			<div id="view" tabindex="-1">
				<div id="controls">
					<span id="repoControl"><span class="unselectable">Repo: </span><div id="repoDropdown" class="dropdown"></div></span>
					<span id="branchControl"><span class="unselectable">Branches: </span><div id="branchDropdown" class="dropdown"></div></span>
					<label id="showRemoteBranchesControl"><input type="checkbox" id="showRemoteBranchesCheckbox" tabindex="-1"><span class="customCheckbox"></span>Show Remote Branches</label>
					<div id="findBtn" title="Find"></div>
					<div id="terminalBtn" title="Open a Terminal for this Repository"></div>
					<div id="settingsBtn" title="Repository Settings"></div>
					<div id="fetchBtn"></div>
					<div id="refreshBtn"></div>
				</div>
				<div id="content">
					<div id="commitGraph"></div>
					<div id="commitTable"></div>
				</div>
				<div id="footer"></div>
			</div>
			<div id="scrollShadow"></div>
			<script nonce="${nonce}">var initialState = ${JSON.stringify(initialState)}, globalState = ${JSON.stringify(globalState)}, workspaceState = ${JSON.stringify(workspaceState)};</script>
			<script nonce="${nonce}" src="${this.getMediaUri('out.min.js')}"></script>
			</body>`;
        }
        else {
            body = `<body class="unableToLoad">
			<h2>Unable to load Git Graph</h2>
			<p class="unableToLoadMessage">No Git repositories were found in the current workspace when it was last scanned by Git Graph.</p>
			<p>If your repositories are in subfolders of the open workspace folder(s), make sure you have set the Git Graph Setting "git-graph.maxDepthOfRepoSearch" appropriately (read the <a href="https://github.com/mhutchie/vscode-git-graph/wiki/Extension-Settings#max-depth-of-repo-search" target="_blank">documentation</a> for more information).</p>
			<p><div id="rescanForReposBtn" class="roundedBtn">Re-scan the current workspace for repositories</div></p>
			<script nonce="${nonce}">(function(){ var api = acquireVsCodeApi(); document.getElementById('rescanForReposBtn').addEventListener('click', function(){ api.postMessage({command: 'rescanForRepos'}); }); })();</script>
			</body>`;
        }
        this.isGraphViewLoaded = numRepos > 0;
        this.loadViewTo = null;
        return `<!DOCTYPE html>
		<html lang="en">
			<head>
				<meta charset="UTF-8">
				<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${standardiseCspSource(this.panel.webview.cspSource)} 'unsafe-inline'; script-src 'nonce-${nonce}'; img-src data:;">
				<meta name="viewport" content="width=device-width, initial-scale=1.0">
				<link rel="stylesheet" type="text/css" href="${this.getMediaUri('out.min.css')}">
				<title>Git Graph</title>
				<style>body{${colorVars}} ${colorParams}</style>
			</head>
			${body}
		</html>`;
    }
    getMediaUri(file) {
        return this.panel.webview.asWebviewUri(this.getUri('media', file));
    }
    getResourcesUri(file) {
        return this.getUri('resources', file);
    }
    getUri(...pathComps) {
        return vscode.Uri.file(path.join(this.extensionPath, ...pathComps));
    }
    respondLoadRepos(repos, loadViewTo) {
        this.sendMessage({
            command: 'loadRepos',
            repos: repos,
            lastActiveRepo: this.extensionState.getLastActiveRepo(),
            loadViewTo: loadViewTo
        });
    }
}
exports.GitGraphView = GitGraphView;
function standardiseCspSource(cspSource) {
    if (cspSource.startsWith('http://') || cspSource.startsWith('https://')) {
        const pathIndex = cspSource.indexOf('/', 8), queryIndex = cspSource.indexOf('?', 8), fragmentIndex = cspSource.indexOf('#', 8);
        let endOfAuthorityIndex = pathIndex;
        if (queryIndex > -1 && (queryIndex < endOfAuthorityIndex || endOfAuthorityIndex === -1))
            endOfAuthorityIndex = queryIndex;
        if (fragmentIndex > -1 && (fragmentIndex < endOfAuthorityIndex || endOfAuthorityIndex === -1))
            endOfAuthorityIndex = fragmentIndex;
        return endOfAuthorityIndex > -1 ? cspSource.substring(0, endOfAuthorityIndex) : cspSource;
    }
    else {
        return cspSource;
    }
}
exports.standardiseCspSource = standardiseCspSource;
//# sourceMappingURL=gitGraphView.js.map