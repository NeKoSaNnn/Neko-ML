// create an array with nodes
const green = '#c2e59c';
const lightgreen = 'rgb(218,253,201)';
const blue = '#64b3f4';
const lightblue = 'rgb(195,220,250)';
const orange = '#FEAC5E';
const gray = '#d7d2cc';
const red = '#E59393';

var options = {
    nodes: {
        value: 1,
        font: {
            size: 13,
            multi: true,
            strokeWidth: 2,
        },
        scaling: {
            min: 35,
            max: 45,
            label: {
                enabled: false,
                maxVisible: 25
            }
        },
    },
    edges: {
        // color: {color: gray},
        smooth: true,
        width: 2
    },
    layout: {
        randomSeed: 1,//配置每次生成的节点位置都一样，参数为数字1、2等
    }
};

var nodes = new vis.DataSet();
var edges = new vis.DataSet();

var edge_nums = 0;
var now_edges = new Set();

var data = {
    nodes: nodes,
    edges: edges
};
var container = document.getElementById('mynetwork');
var network = new vis.Network(container, data, options);

function update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges) {
    // console.log(all_nodes)
    // console.log(server_nodes_to)
    // console.log(client_nodes_to)

    for (let key in all_nodes) {
        nodes.update(all_nodes[key]);
    }
    // server

    if (all_nodes.hasOwnProperty("server")) {
        let server = all_nodes["server"];
        for (let v of server_nodes_to) {
            if (all_nodes.hasOwnProperty(v)) {
                let edge = "from" + server.id + "to" + all_nodes[v].id;
                if (!all_edges.has(edge)) {
                    all_edges.add(edge);
                    edges.add({from: server.id, to: all_nodes[v].id, color: lightblue});
                }
            }
        }
    }
    for (let k in client_nodes_to) {
        if (all_nodes.hasOwnProperty(k)) {
            let client = all_nodes[k];
            if (client_nodes_to.hasOwnProperty(k)) {
                for (let v of client_nodes_to[k]) {
                    if (all_nodes.hasOwnProperty(v)) {
                        let edge = "from" + client.id + "to" + all_nodes[v].id;
                        if (!all_edges.has(edge)) {
                            all_edges.add(edge);
                            edges.add({from: client.id, to: all_nodes[v].id, color: lightgreen});
                        }
                    }
                }
            }
        }
    }
}

function update_node(now_node) {
    nodes.update(now_node);
}

function update_edge(from_node, to_node, all_edges, color) {
    let edge = "from" + from_node.id + "to" + to_node.id;
    if (!all_edges.has(edge)) {
        all_edges.add(edge);
        edges.add({from: from_node.id, to: to_node.id, color: color});
    }
}

function update_client_nodes_to(client_nodes_to, sid, task_id) {
    if (client_nodes_to.hasOwnProperty(sid)) {
        if (!client_nodes_to[sid].has(task_id)) {
            client_nodes_to[sid].add(task_id)
        }
    } else {
        client_nodes_to[sid] = new Set();
        if (!client_nodes_to[sid].has(task_id)) {
            client_nodes_to[sid].add(task_id)
        }
    }
}

function update_server_nodes_to(server_nodes_to, task_id) {
    if (!server_nodes_to.has(task_id)) {
        server_nodes_to.add(task_id)
    }
}

$(document).ready(function () {
    setTimeout(function () {
        let name_space = "/ui";
        var sio = io.connect(location.protocol + "//" + document.domain + ":" + location.port + name_space);
        console.log(sio)
        sio.pingTimeout = 36000000;
        sio.pingInterval = 300000;
        // console.log(sio)
        // console.log(location.protocol + "//" + document.domain + ":" + location.port + name_space)
        // console.log(sio)
        // console.log(sio.heartbeatTimeout)
        var all_nodes = Array();
        var server_nodes_to = new Set();
        var client_nodes_to = new Set();
        var all_edges = new Set();
        var now_nodes = 0;
        var now_client = 0;

        window.onbeforeunload = function (e) {
            e.preventDefault();
            sio.onclose = function () {
            }
            sio.close();
            return "close";
        }

        // window.onunload = () => {
        //     sio.onclose = function () {
        //     }
        //     sio.close();
        //     return "close";
        // }

        sio.on("connect", function () {
            console.log("connect");
        });

        sio.on("disconnect", function () {
            console.log("disconnect");
            sio.close();
        });

        sio.on("ui_disconnect", function () {
            console.log("disconnect");
            sio.close();
        });

        sio.on("s_connect", function () {
            console.log("server_connect")
            now_nodes++;
            all_nodes["server"] = {
                "id": now_nodes,
                "label": "<b>server</b>",
                "value": 1,
                "color": blue,
                "shape": "hexagon",
            };
            server_nodes_to = new Set();
            update_node(all_nodes["server"]);
            // update_network(all_nodes, server_nodes_to, client_nodes_to);
        });
        sio.on("c_connect", function (res) {
            console.log("client_connect")
            let sid = res["sid"];
            if (all_nodes.hasOwnProperty(sid)) {
                all_nodes[sid].color = {
                    "border": gray,
                    "background": "#F8F8F8",
                };
            } else {
                now_client++;
                now_nodes++;
                let now_label = "<b>client-" + now_client + "</b>";
                all_nodes[sid] = {
                    "id": now_nodes,
                    "label": now_label,
                    "value": 1,
                    "color": {
                        "border": gray,
                        "background": "#F8F8F8"
                    },
                    "shape": "dot",
                };
                client_nodes_to[sid] = new Set();
            }
            update_node(all_nodes[sid]);
            if (!all_nodes.hasOwnProperty("server")) {
                now_nodes++;
                all_nodes["server"] = {
                    "id": now_nodes,
                    "label": "<b>server</b>",
                    "value": 1,
                    "color": blue,
                    "shape": "hexagon",
                };
                server_nodes_to = new Set();
                update_node(all_nodes["server"]);
            }
            update_client_nodes_to(client_nodes_to, sid, "server")
            update_edge(all_nodes[sid], all_nodes["server"], all_edges, lightblue);

            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        });
        sio.on("c_wakeup", async function (res) {
            console.log("wakeup")
            let sid = res["sid"];
            if (all_nodes.hasOwnProperty(sid)) {
                all_nodes[sid].color = green;
            } else {
                now_client++;
                now_nodes++;
                let now_label = "<b>client-" + now_client + "</b>";
                all_nodes[sid] = {
                    "id": now_nodes,
                    "label": now_label,
                    "value": 1,
                    "color": green,
                    "shape": "dot",
                };
                client_nodes_to[sid] = new Set();
            }
            update_node(all_nodes[sid]);
            if (!all_nodes.hasOwnProperty("server")) {
                now_nodes++;
                all_nodes["server"] = {
                    "id": now_nodes,
                    "label": "<b>server</b>",
                    "value": 1,
                    "color": blue,
                    "shape": "hexagon",
                };
                server_nodes_to = new Set();
                update_node(all_nodes["server"]);
            }
            update_client_nodes_to(client_nodes_to, sid, "server")
            update_edge(all_nodes[sid], all_nodes["server"], all_edges, lightblue);

            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        })
        sio.on("c_reconnect", function (res) {
            console.log("client_reconnect");
        });
        sio.on("c_disconnect", function (res) {
            console.log("client_disconnect");
            let sid = res["sid"];
            if (all_nodes.hasOwnProperty(sid)) {
                all_nodes[sid].color = red;
                update_node(all_nodes[sid]);
            }
            for (let child of client_nodes_to[sid]) {
                if (child !== "server") {
                    all_nodes[child].color = red;
                    update_node(all_nodes[child]);
                }
            }
            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        });
        sio.on("c_check_resource", function (res) {
            console.log("check_resource")
            let sid = res["sid"];
            let task_id = "check_resource_" + sid;
            if (!all_nodes.hasOwnProperty(sid)) {
                now_nodes++;
                now_client++;
                let now_label = "<b>client-" + now_client + "</b>";
                all_nodes[sid] = {
                    "id": now_nodes,
                    "label": now_label,
                    "value": 1,
                    "color": green,
                    "shape": "dot",
                };
                update_node(all_nodes[sid]);
            }
            let now_client_id = all_nodes[sid].label;
            if (all_nodes.hasOwnProperty(task_id)) {
                all_nodes[task_id].color = green;
                all_nodes[task_id].label = now_client_id + "\n<i>check_resource</i>";
            } else {
                now_nodes++;
                let now_label = now_client_id + "\n<i>check_resource</i>";
                all_nodes[task_id] = {
                    "id": now_nodes,
                    "label": now_label,
                    "value": 2,
                    "color": green,
                    "shape": "box",
                };
            }
            update_node(all_nodes[task_id]);
            if (!all_nodes.hasOwnProperty("server")) {
                now_nodes++;
                all_nodes["server"] = {
                    "id": now_nodes,
                    "label": "<b>server</b>",
                    "value": 1,
                    "color": blue,
                    "shape": "hexagon",
                };
                server_nodes_to = new Set();
                update_node(all_nodes["server"]);
            }
            update_client_nodes_to(client_nodes_to, sid, "server")
            update_client_nodes_to(client_nodes_to, sid, task_id);

            update_edge(all_nodes[sid], all_nodes["server"], all_edges, lightblue);
            update_edge(all_nodes[sid], all_nodes[task_id], all_edges, lightgreen);

            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        });
        sio.on("c_check_resource_complete", function (res) {
            console.log("check_resource_complete")
            let sid = res["sid"];
            let task_id = "check_resource_" + sid;
            if (!all_nodes.hasOwnProperty(sid)) {
                now_nodes++;
                now_client++;
                let now_label = "<b>client-" + now_client + "</b>";
                all_nodes[sid] = {
                    "id": now_nodes,
                    "label": now_label,
                    "value": 1,
                    "color": green,
                    "shape": "dot",
                };
                update_node(all_nodes[sid]);
            }
            let now_client_id = all_nodes[sid].label;
            if (all_nodes.hasOwnProperty(task_id)) {
                all_nodes[task_id].color = {
                    "border": gray,
                    "background": "#F8F8F8"
                };
                all_nodes[task_id].label = now_client_id + "\n<i>check_resource complete</i>";
            } else {
                now_nodes++;
                let now_label = now_client_id + "\n<i>check_resource complete</i>";
                all_nodes[task_id] = {
                    "id": now_nodes,
                    "label": now_label,
                    "value": 2,
                    "color": {
                        "border": gray,
                        "background": "#F8F8F8"
                    },
                    "shape": "box",
                };
            }
            update_node(all_nodes[task_id]);
            if (!all_nodes.hasOwnProperty("server")) {
                now_nodes++;
                all_nodes["server"] = {
                    "id": now_nodes,
                    "label": "<b>server</b>",
                    "value": 1,
                    "color": blue,
                    "shape": "hexagon",
                };
                server_nodes_to = new Set();
                update_node(all_nodes["server"]);
            }
            update_client_nodes_to(client_nodes_to, sid, "server")
            update_client_nodes_to(client_nodes_to, sid, task_id);

            update_edge(all_nodes[sid], all_nodes["server"], all_edges, lightblue);
            update_edge(all_nodes[sid], all_nodes[task_id], all_edges, lightgreen);

            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        });
        sio.on("c_train", function (res) {
            console.log("train")
            let sid = res["sid"];
            let gep = res["gep"];
            let task_id = "train_" + sid;
            if (!all_nodes.hasOwnProperty(sid)) {
                now_nodes++;
                now_client++;
                let now_label = "<b>client-" + now_client + "</b>";
                all_nodes[sid] = {
                    "id": now_nodes,
                    "label": now_label,
                    "value": 1,
                    "color": green,
                    "shape": "dot",
                };
                update_node(all_nodes[sid]);
            }
            let now_client_id = all_nodes[sid].label;
            if (all_nodes.hasOwnProperty(task_id)) {
                all_nodes[task_id].color = green;
                all_nodes[task_id].label = now_client_id + "\n<i>train: </i><b>" + gep + "</b>";
            } else {
                now_nodes++;
                let now_label = now_client_id + "\n<i>train: </i><b>" + gep + "</b>";
                all_nodes[task_id] = {
                    "id": now_nodes,
                    "label": now_label,
                    "value": 2,
                    "color": green,
                    "shape": "box",
                    "title": "train",
                };
            }
            update_node(all_nodes[task_id]);
            if (!all_nodes.hasOwnProperty("server")) {
                now_nodes++;
                all_nodes["server"] = {
                    "id": now_nodes,
                    "label": "<b>server</b>",
                    "value": 1,
                    "color": blue,
                    "shape": "hexagon",
                };
                server_nodes_to = new Set();
                update_node(all_nodes["server"]);
            }
            update_client_nodes_to(client_nodes_to, sid, "server")
            update_client_nodes_to(client_nodes_to, sid, task_id);

            update_edge(all_nodes[sid], all_nodes["server"], all_edges, lightblue);
            update_edge(all_nodes[sid], all_nodes[task_id], all_edges, lightgreen);

            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        });
        sio.on("ui_train_process", function (res) {
            // console.log("train_process")
            let sid = res["sid"];
            let gep = res["gep"];
            let process = res["process"];
            let task_id = "train_" + sid;
            if (!all_nodes.hasOwnProperty(sid)) {
                now_nodes++;
                now_client++;
                let now_label = "<b>client-" + now_client + "</b>";
                all_nodes[sid] = {
                    "id": now_nodes,
                    "label": now_label,
                    "value": 1,
                    "color": green,
                    "shape": "dot",
                };
                update_node(all_nodes[sid]);
            }
            let now_client_id = all_nodes[sid].label;
            if (all_nodes.hasOwnProperty(task_id)) {
                all_nodes[task_id].color = green;
                all_nodes[task_id].label = now_client_id + "\n<i>train: </i><b>" + gep + " [" + process + "]</b>";
            } else {
                now_nodes++;
                let now_label = now_client_id + "\n<i>train: </i><b>" + gep + " [" + process + "]</b>";
                all_nodes[task_id] = {
                    "id": now_nodes,
                    "label": now_label,
                    "value": 2,
                    "color": green,
                    "shape": "box",
                    "title": "train",
                };
            }
            update_node(all_nodes[task_id]);
            if (!all_nodes.hasOwnProperty("server")) {
                now_nodes++;
                all_nodes["server"] = {
                    "id": now_nodes,
                    "label": "<b>server</b>",
                    "value": 1,
                    "color": blue,
                    "shape": "hexagon",
                };
                server_nodes_to = new Set();
                update_node(all_nodes["server"]);
            }
            update_client_nodes_to(client_nodes_to, sid, "server")
            update_client_nodes_to(client_nodes_to, sid, task_id);

            update_edge(all_nodes[sid], all_nodes["server"], all_edges, lightblue);
            update_edge(all_nodes[sid], all_nodes[task_id], all_edges, lightgreen);

            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        });
        sio.on("c_train_complete", function (res) {
            console.log("train_complete")
            let sid = res["sid"];
            let gep = res["gep"];
            let task_id = "train_" + sid;
            if (!all_nodes.hasOwnProperty(sid)) {
                now_nodes++;
                now_client++;
                let now_label = "<b>client-" + now_client + "</b>";
                all_nodes[sid] = {
                    "id": now_nodes,
                    "label": now_label,
                    "value": 1,
                    "color": green,
                    "shape": "dot",
                };
                update_node(all_nodes[sid]);
            }
            let now_client_id = all_nodes[sid].label;
            if (all_nodes.hasOwnProperty(task_id)) {
                all_nodes[task_id].color = {
                    "border": gray,
                    "background": "#F8F8F8"
                };
                all_nodes[task_id].label = now_client_id + "\n<i>train: " + gep + " complete</i>";
            } else {
                now_nodes++;
                let now_label = now_client_id + "\n<i>train: " + gep + " complete</i>";
                all_nodes[task_id] = {
                    "id": now_nodes,
                    "label": now_label,
                    "value": 2,
                    "color": {
                        "border": gray,
                        "background": "#F8F8F8"
                    },
                    "shape": "box",
                    "title": "train",
                };
            }
            update_node(all_nodes[task_id]);
            if (!all_nodes.hasOwnProperty("server")) {
                now_nodes++;
                all_nodes["server"] = {
                    "id": now_nodes,
                    "label": "<b>server</b>",
                    "value": 1,
                    "color": blue,
                    "shape": "hexagon",
                };
                server_nodes_to = new Set();
                update_node(all_nodes["server"]);
            }
            update_client_nodes_to(client_nodes_to, sid, "server")
            update_client_nodes_to(client_nodes_to, sid, task_id);

            update_edge(all_nodes[sid], all_nodes["server"], all_edges, lightblue);
            update_edge(all_nodes[sid], all_nodes[task_id], all_edges, lightgreen);

            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        });
        sio.on("s_train_aggre", function (res) {
            console.log("train_aggre")
            let gep = res["gep"];
            let task_id = "train_aggre";
            if (all_nodes.hasOwnProperty(task_id)) {
                all_nodes[task_id].color = blue;
                all_nodes[task_id].label = "<b>server</b>\n<i>train_aggre:</i><b>" + gep + "</b>";
            } else {
                now_nodes++;
                all_nodes[task_id] = {
                    "id": now_nodes,
                    "label": "<b>server</b>\n<i>train_aggre:</i><b>" + gep + "</b>",
                    "value": 3,
                    "color": blue,
                    "shape": "box",
                };
            }
            update_node(all_nodes[task_id]);
            if (!all_nodes.hasOwnProperty("server")) {
                now_nodes++;
                all_nodes["server"] = {
                    "id": now_nodes,
                    "label": "<b>server</b>",
                    "value": 1,
                    "color": blue,
                    "shape": "hexagon",
                };
                server_nodes_to = new Set();
                update_node(all_nodes["server"]);
            }
            update_server_nodes_to(server_nodes_to, task_id);
            update_edge(all_nodes["server"], all_nodes[task_id], all_edges, lightblue);

            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        });
        sio.on("s_train_aggre_complete", function (res) {
            console.log("train_aggre_complete")
            let gep = res["gep"];
            let task_id = "train_aggre";
            if (all_nodes.hasOwnProperty(task_id)) {
                all_nodes[task_id].color = {
                    "border": gray,
                    "background": "#F8F8F8"
                };
                all_nodes[task_id].label = "<b>server</b>\n<i>train_aggre: " + gep + " complete</i>";
            } else {
                now_nodes++;
                all_nodes[task_id] = {
                    "id": now_nodes,
                    "label": "<b>server</b>\n<i>train_aggre: " + gep + " complete</i>",
                    "value": 3,
                    "color": {
                        "border": gray,
                        "background": "#F8F8F8"
                    },
                    "shape": "box",
                };
            }
            update_node(all_nodes[task_id]);
            if (!all_nodes.hasOwnProperty("server")) {
                now_nodes++;
                all_nodes["server"] = {
                    "id": now_nodes,
                    "label": "<b>server</b>",
                    "value": 1,
                    "color": blue,
                    "shape": "hexagon",
                };
                server_nodes_to = new Set();
                update_node(all_nodes["server"]);
            }
            update_server_nodes_to(server_nodes_to, task_id);
            update_edge(all_nodes["server"], all_nodes[task_id], all_edges, lightblue);

            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        });
        sio.on("c_eval", function (res) {
            console.log("eval")
            let sid = res["sid"];
            let gep = res["gep"];
            let task_id = "eval_" + sid;
            if (!all_nodes.hasOwnProperty(sid)) {
                now_nodes++;
                now_client++;
                let now_label = "<b>client-" + now_client + "</b>";
                all_nodes[sid] = {
                    "id": now_nodes,
                    "label": now_label,
                    "value": 1,
                    "color": green,
                    "shape": "dot",
                };
                update_node(all_nodes[sid]);
            }
            let now_client_id = all_nodes[sid].label;
            if (all_nodes.hasOwnProperty(task_id)) {
                all_nodes[task_id].color = green;
                all_nodes[task_id].label = now_client_id + "\n<i>eval: </i><b>" + gep + "</b>";
            } else {
                now_nodes++;
                let now_label = now_client_id + "\n<i>eval: </i><b>" + gep + "</b>";
                all_nodes[task_id] = {
                    "id": now_nodes,
                    "label": now_label,
                    "value": 2,
                    "color": green,
                    "shape": "box",
                };
            }
            update_node(all_nodes[task_id]);
            if (!all_nodes.hasOwnProperty("server")) {
                now_nodes++;
                all_nodes["server"] = {
                    "id": now_nodes,
                    "label": "<b>server</b>",
                    "value": 1,
                    "color": blue,
                    "shape": "hexagon",
                };
                server_nodes_to = new Set();
                update_node(all_nodes["server"]);
            }
            update_client_nodes_to(client_nodes_to, sid, "server")
            update_client_nodes_to(client_nodes_to, sid, task_id);

            update_edge(all_nodes[sid], all_nodes["server"], all_edges, lightblue);
            update_edge(all_nodes[sid], all_nodes[task_id], all_edges, lightgreen);

            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        });
        sio.on("ui_eval_process", function (res) {
            // console.log("eval_process")
            let sid = res["sid"];
            let gep = res["gep"];
            let process = res["process"];
            let type = res["type"];
            let task_id = "eval_" + sid;
            if (!all_nodes.hasOwnProperty(sid)) {
                now_nodes++;
                now_client++;
                let now_label = "<b>client-" + now_client + "</b>";
                all_nodes[sid] = {
                    "id": now_nodes,
                    "label": now_label,
                    "value": 1,
                    "color": green,
                    "shape": "dot",
                };
                update_node(all_nodes[sid]);
            }
            let now_client_id = all_nodes[sid].label;
            if (all_nodes.hasOwnProperty(task_id)) {
                all_nodes[task_id].color = green;
                all_nodes[task_id].label = now_client_id + "\n<i>eval(" + type + "): </i><b>" + gep + " [" + process + "]</b>";
            } else {
                now_nodes++;
                let now_label = now_client_id + "\n<i>eval(" + type + "): </i><b>" + gep + " [" + process + "]</b>";
                all_nodes[task_id] = {
                    "id": now_nodes,
                    "label": now_label,
                    "value": 2,
                    "color": green,
                    "shape": "box",
                };
            }
            update_node(all_nodes[task_id]);
            if (!all_nodes.hasOwnProperty("server")) {
                now_nodes++;
                all_nodes["server"] = {
                    "id": now_nodes,
                    "label": "<b>server</b>",
                    "value": 1,
                    "color": blue,
                    "shape": "hexagon",
                };
                server_nodes_to = new Set();
                update_node(all_nodes["server"]);
            }
            update_client_nodes_to(client_nodes_to, sid, "server")
            update_client_nodes_to(client_nodes_to, sid, task_id);

            update_edge(all_nodes[sid], all_nodes["server"], all_edges, lightblue);
            update_edge(all_nodes[sid], all_nodes[task_id], all_edges, lightgreen);

            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        });
        sio.on("c_eval_complete", function (res) {
            console.log("eval_complete")
            let sid = res["sid"];
            let gep = res["gep"];
            let task_id = "eval_" + sid;
            if (!all_nodes.hasOwnProperty(sid)) {
                now_nodes++;
                now_client++;
                let now_label = "<b>client-" + now_client + "</b>";
                all_nodes[sid] = {
                    "id": now_nodes,
                    "label": now_label,
                    "value": 1,
                    "color": green,
                    "shape": "dot",
                };
                update_node(all_nodes[sid]);
            }
            let now_client_id = all_nodes[sid].label;
            if (all_nodes.hasOwnProperty(task_id)) {
                all_nodes[task_id].color = {
                    "border": gray,
                    "background": "#F8F8F8"
                };
                all_nodes[task_id].label = now_client_id + "\n<i>eval: " + gep + " complete</i>";
            } else {
                now_nodes++;
                let now_label = now_client_id + "\n<i>eval: " + gep + " complete</i>";
                all_nodes[task_id] = {
                    "id": now_nodes,
                    "label": now_label,
                    "value": 2,
                    "color": {
                        "border": gray,
                        "background": "#F8F8F8"
                    },
                    "shape": "box",
                };
            }
            update_node(all_nodes[task_id]);
            if (!all_nodes.hasOwnProperty("server")) {
                now_nodes++;
                all_nodes["server"] = {
                    "id": now_nodes,
                    "label": "<b>server</b>",
                    "value": 1,
                    "color": blue,
                    "shape": "hexagon",
                };
                server_nodes_to = new Set();
                update_node(all_nodes["server"]);
            }
            update_client_nodes_to(client_nodes_to, sid, "server")
            update_client_nodes_to(client_nodes_to, sid, task_id);

            update_edge(all_nodes[sid], all_nodes["server"], all_edges, lightblue);
            update_edge(all_nodes[sid], all_nodes[task_id], all_edges, lightgreen);

            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        });
        sio.on("s_eval_aggre", function (res) {
            console.log("eval_aggre")
            let gep = res["gep"];
            let task_id = "eval_aggre";
            if (all_nodes.hasOwnProperty(task_id)) {
                all_nodes[task_id].color = blue;
                all_nodes[task_id].label = "<b>server</b>\n<i>eval_aggre:</i><b>" + gep + "</b>";
            } else {
                now_nodes++;
                all_nodes[task_id] = {
                    "id": now_nodes,
                    "label": "<b>server</b>\n<i>eval_aggre:</i><b>" + gep + "</b>",
                    "value": 3,
                    "color": blue,
                    "shape": "box",
                }
            }
            update_node(all_nodes[task_id]);
            if (!all_nodes.hasOwnProperty("server")) {
                now_nodes++;
                all_nodes["server"] = {
                    "id": now_nodes,
                    "label": "<b>server</b>",
                    "value": 1,
                    "color": blue,
                    "shape": "hexagon",
                };
                server_nodes_to = new Set();
                update_node(all_nodes["server"]);
            }
            update_server_nodes_to(server_nodes_to, task_id);
            update_edge(all_nodes["server"], all_nodes[task_id], all_edges, lightblue);

            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        });
        sio.on("s_eval_aggre_complete", function (res) {
            console.log("eval_aggre_complete")
            let gep = res["gep"];
            let task_id = "eval_aggre";
            if (all_nodes.hasOwnProperty(task_id)) {
                all_nodes[task_id].color = {
                    "border": gray,
                    "background": "#F8F8F8"
                };
                all_nodes[task_id].label = "<b>server</b>\n<i>eval_aggre: " + gep + " complete</i>";
            } else {
                now_nodes++;
                all_nodes[task_id] = {
                    "id": now_nodes,
                    "label": "<b>server</b>\n<i>eval_aggre: " + gep + " complete</i>",
                    "value": 3,
                    "color": {
                        "border": gray,
                        "background": "#F8F8F8"
                    },
                    "shape": "box",
                }
            }
            update_node(all_nodes[task_id]);
            if (!all_nodes.hasOwnProperty("server")) {
                now_nodes++;
                all_nodes["server"] = {
                    "id": now_nodes,
                    "label": "<b>server</b>",
                    "value": 1,
                    "color": blue,
                    "shape": "hexagon",
                };
                server_nodes_to = new Set();
                update_node(all_nodes["server"]);
            }
            update_server_nodes_to(server_nodes_to, task_id);
            update_edge(all_nodes["server"], all_nodes[task_id], all_edges, lightblue);

            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        });
        sio.on("s_summary", function () {
            console.log("summary")
            let task_id = "summary";
            if (all_nodes.hasOwnProperty(task_id)) {
                all_nodes[task_id].color = blue;
                all_nodes[task_id].label = "<b>server</b>\n<i>summary</i>";
            } else {
                now_nodes++;
                all_nodes[task_id] = {
                    "id": now_nodes,
                    "label": "<b>server</b>\n<i>summary</i>",
                    "value": 3,
                    "color": blue,
                    "shape": "box",
                }
            }
            update_node(all_nodes[task_id]);
            if (!all_nodes.hasOwnProperty("server")) {
                now_nodes++;
                all_nodes["server"] = {
                    "id": now_nodes,
                    "label": "<b>server</b>",
                    "value": 1,
                    "color": blue,
                    "shape": "hexagon",
                };
                server_nodes_to = new Set();
                update_node(all_nodes["server"]);
            }
            update_server_nodes_to(server_nodes_to, task_id);
            update_edge(all_nodes["server"], all_nodes[task_id], all_edges, lightblue);

            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        });
        sio.on("s_summary_complete", function () {
            console.log("summary_complete")
            let task_id = "summary";
            if (all_nodes.hasOwnProperty(task_id)) {
                all_nodes[task_id].color = {
                    "border": gray,
                    "background": "#F8F8F8"
                };
                all_nodes[task_id].label = "<b>server</b>\n<i>summary complete</i>";
            } else {
                now_nodes++;
                all_nodes[task_id] = {
                    "id": now_nodes,
                    "label": "<b>server</b>\n<i>summary complete</i>",
                    "value": 2,
                    "color": {
                        "border": gray,
                        "background": "#F8F8F8"
                    },
                    "shape": "box",
                }
            }
            update_node(all_nodes[task_id]);
            if (!all_nodes.hasOwnProperty("server")) {
                now_nodes++;
                all_nodes["server"] = {
                    "id": now_nodes,
                    "label": "<b>server</b>",
                    "value": 1,
                    "color": blue,
                    "shape": "hexagon",
                };
                server_nodes_to = new Set();
                update_node(all_nodes["server"]);
            }
            update_server_nodes_to(server_nodes_to, task_id);
            update_edge(all_nodes["server"], all_nodes[task_id], all_edges, lightblue);

            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        });
        sio.on("c_fin", function (res) {
            console.log("client_fin")
            let sid = res["sid"];
            if (all_nodes.hasOwnProperty(sid)) {
                all_nodes[sid].color = {
                    "border": gray,
                    "background": "#F8F8F8"
                };
                update_node(all_nodes[sid]);
            }
            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        });
        sio.on("s_fin", function () {
            console.log("server_fin")
            if (all_nodes.hasOwnProperty("server")) {
                all_nodes["server"].color = {
                    "border": gray,
                    "background": "#F8F8F8"
                };
                update_node(all_nodes["server"]);
            }
            // update_network(all_nodes, server_nodes_to, client_nodes_to, all_edges);
        });
    }, 2000);
});