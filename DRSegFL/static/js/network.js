// create an array with nodes
const green = '#94BFA2';
const orange = '#F9A657';
const gray = '#EAEAEB';
const red = '#E59393';

var options = {
    nodes: {
        value: 1,
        shape: 'dot',
        font: {
            size: 12,
            multi: true,
            strokeWidth: 3,
        },
        scaling: {
            min: 10,
            max: 40,
            label: {
                enabled: false,
                maxVisible: 5
            }
        },
    },
    edges: {
        color: {color: gray},
        smooth: false,
        width: 3
    },
    layout: {
        randomSeed: 1,//配置每次生成的节点位置都一样，参数为数字1、2等
    }
};


function update_network(all_nodes, server_nodes_to, client_nodes_to) {
    console.log(all_nodes)
    console.log(server_nodes_to)
    console.log(client_nodes_to)
    var now_nodes = Array();
    var now_edges = Array();

    for (let key in all_nodes) {
        now_nodes.push(all_nodes[key]);
    }
    // server

    if (all_nodes.hasOwnProperty("server")) {
        let server = all_nodes["server"];
        for (let v of server_nodes_to) {
            if (all_nodes.hasOwnProperty(v)) {
                now_edges.push({from: server.id, to: all_nodes[v].id});
            }
        }
    }
    for (let k in client_nodes_to) {
        if (all_nodes.hasOwnProperty(k)) {
            let client = all_nodes[k];
            if (client_nodes_to.hasOwnProperty(k)) {
                for (let v of client_nodes_to[k]) {
                    if (all_nodes.hasOwnProperty(v)) {
                        now_edges.push({from: client.id, to: all_nodes[v].id});
                    }
                }
            }
        }
    }
    var nodes = new vis.DataSet(now_nodes);
    var edges = new vis.DataSet(now_edges);

    var data = {
        nodes: nodes,
        edges: edges
    };
    var container = document.getElementById('mynetwork');
    var network = new vis.Network(container, data, options);
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

let name_space = "/ui";
const socket = io.connect(location.protocol + "//" + document.domain + ":" + location.port + name_space);
console.log(location.protocol + "//" + document.domain + ":" + location.port + name_space)
var all_nodes = Array();
var server_nodes_to = new Set();
var client_nodes_to = new Set();
var now_nodes = 0;
var now_client = 0;
socket.on("s_connect", function () {
    console.log("server_connect")
    now_nodes++;
    all_nodes["server"] = {
        "id": now_nodes,
        "label": "<b>server</b>",
        "value": 1,
        "color": green,
    };
    server_nodes_to = new Set();
    update_network(all_nodes, server_nodes_to, client_nodes_to);
});
socket.on("c_connect", function (res) {
    console.log("client_connect")
    let sid = res["sid"];
    if (all_nodes.hasOwnProperty(sid)) {
        all_nodes[sid].color = {
            "border": gray,
            "background": "#F8F8F8"
        };
    } else {
        now_client++;
        now_nodes++;
        let now_label = "<b>client_" + now_client + "</b>";
        all_nodes[sid] = {
            "id": now_nodes,
            "label": now_label,
            "value": 1,
            "color": {
                "border": gray,
                "background": "#F8F8F8"
            },
        };
        client_nodes_to[sid] = new Set();
    }
    if (!all_nodes.hasOwnProperty("server")) {
        now_nodes++;
        all_nodes["server"] = {
            "id": now_nodes,
            "label": "<b>server</b>",
            "value": 1,
            "color": green,
        };
        server_nodes_to = new Set();
    }
    update_client_nodes_to(client_nodes_to, sid, "server")
    update_network(all_nodes, server_nodes_to, client_nodes_to);
});
socket.on("c_wakeup", async function (res) {
    console.log("wakeup")
    let sid = res["sid"];
    if (all_nodes.hasOwnProperty(sid)) {
        all_nodes[sid].color = green;
    } else {
        now_client++;
        now_nodes++;
        let now_label = "<b>client_" + now_client + "</b>";
        all_nodes[sid] = {
            "id": now_nodes,
            "label": now_label,
            "value": 1,
            "color": green,
        };
        client_nodes_to[sid] = new Set();
    }
    if (!all_nodes.hasOwnProperty("server")) {
        now_nodes++;
        all_nodes["server"] = {
            "id": now_nodes,
            "label": "<b>server</b>",
            "value": 1,
            "color": green,
        };
        server_nodes_to = new Set();
    }
    update_client_nodes_to(client_nodes_to, sid, "server")
    update_client_nodes_to(client_nodes_to, sid, "server");
    update_network(all_nodes, server_nodes_to, client_nodes_to);
})
socket.on("c_reconnect", function (res) {

});
socket.on("c_disconnect", function (res) {
    console.log("client_disconnect")
    let sid = res["sid"];
    if (all_nodes.hasOwnProperty(sid)) {
        all_nodes[sid].color = red;
    }
    update_network(all_nodes, server_nodes_to, client_nodes_to);
});
socket.on("c_check_resource", function (res) {
    console.log("check_resource")
    let sid = res["sid"];
    let task_id = "check_resource_" + sid;
    if (!all_nodes.hasOwnProperty(sid)) {
        now_nodes++;
        now_client++;
        let now_label = "<b>client_" + now_client + "</b>";
        all_nodes[sid] = {
            "id": now_nodes,
            "label": now_label,
            "value": 1,
            "color": green,
        };
    }
    if (all_nodes.hasOwnProperty(task_id)) {
        all_nodes[task_id].color = green;
        all_nodes[task_id].label = now_client_id + "\n<i>check_resource</i>";
    } else {
        now_nodes++;
        let now_client_id = all_nodes[sid].label;
        let now_label = now_client_id + "\n<i>check_resource</i>";
        all_nodes[task_id] = {
            "id": now_nodes,
            "label": now_label,
            "value": 1,
            "color": green,
        };
    }
    if (!all_nodes.hasOwnProperty("server")) {
        now_nodes++;
        all_nodes["server"] = {
            "id": now_nodes,
            "label": "<b>server</b>",
            "value": 1,
            "color": green,
        };
        server_nodes_to = new Set();
    }
    update_client_nodes_to(client_nodes_to, sid, "server")
    update_client_nodes_to(client_nodes_to, sid, task_id);
    update_network(all_nodes, server_nodes_to, client_nodes_to);
});
socket.on("c_check_resource_complete", function (res) {
    console.log("check_resource_complete")
    let sid = res["sid"];
    let task_id = "check_resource_" + sid;
    if (!all_nodes.hasOwnProperty(sid)) {
        now_nodes++;
        now_client++;
        let now_label = "<b>client_" + now_client + "</b>";
        all_nodes[sid] = {
            "id": now_nodes,
            "label": now_label,
            "value": 1,
            "color": green,
        };
    }
    let now_client_id = all_nodes[sid].label;
    if (all_nodes.hasOwnProperty(task_id)) {
        all_nodes[task_id].color = {
            "border": gray,
            "background": "#F8F8F8"
        };
        all_nodes[task_id].label = now_client_id + "\n<i>check_resource</i>\n<b>complete</b>";
    } else {
        now_nodes++;
        let now_label = now_client_id + "\n<i>check_resource</i>\n<b>complete</b>";
        all_nodes[task_id] = {
            "id": now_nodes,
            "label": now_label,
            "value": 1,
            "color": {
                "border": gray,
                "background": "#F8F8F8"
            },
        };
    }
    if (!all_nodes.hasOwnProperty("server")) {
        now_nodes++;
        all_nodes["server"] = {
            "id": now_nodes,
            "label": "<b>server</b>",
            "value": 1,
            "color": green,
        };
        server_nodes_to = new Set();
    }
    update_client_nodes_to(client_nodes_to, sid, "server")
    update_client_nodes_to(client_nodes_to, sid, task_id);
    update_network(all_nodes, server_nodes_to, client_nodes_to);
});
socket.on("c_train", function (res) {
    console.log("train")
    let sid = res["sid"];
    let gep = res["gep"];
    let task_id = "train_" + sid;
    if (!all_nodes.hasOwnProperty(sid)) {
        now_nodes++;
        now_client++;
        let now_label = "<b>client_" + now_client + "</b>";
        all_nodes[sid] = {
            "id": now_nodes,
            "label": now_label,
            "value": 1,
            "color": green,
        };
    }
    let now_client_id = all_nodes[sid].label;
    if (all_nodes.hasOwnProperty(task_id)) {
        all_nodes[task_id].color = green;
        all_nodes[task_id].label = now_client_id + "\n<i>train:</i><b>" + gep + "</b>";
    } else {
        now_nodes++;
        let now_label = now_client_id + "\n<i>train:</i><b>" + gep + "</b>";
        all_nodes[task_id] = {
            "id": now_nodes,
            "label": now_label,
            "value": 1,
            "color": green,
        };
    }
    if (!all_nodes.hasOwnProperty("server")) {
        now_nodes++;
        all_nodes["server"] = {
            "id": now_nodes,
            "label": "<b>server</b>",
            "value": 1,
            "color": green,
        };
        server_nodes_to = new Set();
    }
    update_client_nodes_to(client_nodes_to, sid, "server")
    update_client_nodes_to(client_nodes_to, sid, task_id);
    update_network(all_nodes, server_nodes_to, client_nodes_to);
});
socket.on("c_train_complete", function (res) {
    console.log("train_complete")
    let sid = res["sid"];
    let gep = res["gep"];
    let task_id = "train_" + sid;
    if (!all_nodes.hasOwnProperty(sid)) {
        now_nodes++;
        now_client++;
        let now_label = "<b>client_" + now_client + "</b>";
        all_nodes[sid] = {
            "id": now_nodes,
            "label": now_label,
            "value": 1,
            "color": green,
        };
    }
    let now_client_id = all_nodes[sid].label;
    if (all_nodes.hasOwnProperty(task_id)) {
        all_nodes[task_id].color = {
            "border": gray,
            "background": "#F8F8F8"
        };
        all_nodes[task_id].label = now_client_id + "\n<i>train:</i><b>" + gep + "</b>\n<b>complete</b>";
    } else {
        now_nodes++;
        let now_label = now_client_id + "\n<i>train:</i><b>" + gep + "</b>\n<b>complete</b>";
        all_nodes[task_id] = {
            "id": now_nodes,
            "label": now_label,
            "value": 1,
            "color": {
                "border": gray,
                "background": "#F8F8F8"
            },
        };
    }
    if (!all_nodes.hasOwnProperty("server")) {
        now_nodes++;
        all_nodes["server"] = {
            "id": now_nodes,
            "label": "<b>server</b>",
            "value": 1,
            "color": green,
        };
        server_nodes_to = new Set();
    }
    update_client_nodes_to(client_nodes_to, sid, "server")
    update_client_nodes_to(client_nodes_to, sid, task_id);
    update_network(all_nodes, server_nodes_to, client_nodes_to);
});
socket.on("s_train_aggre", function (res) {
    console.log("train_aggre")
    let gep = res["gep"];
    let task_id = "train_aggre";
    if (all_nodes.hasOwnProperty(task_id)) {
        all_nodes[task_id].color = green;
        all_nodes[task_id].label = "<b>server</b>\n<i>train_aggre:</i><b>" + gep + "</b>";
    } else {
        now_nodes++;
        all_nodes[task_id] = {
            "id": now_nodes,
            "label": "<b>server</b>\n<i>train_aggre:</i><b>" + gep + "</b>",
            "value": 1,
            "color": green,
        };
    }
    if (!all_nodes.hasOwnProperty("server")) {
        now_nodes++;
        all_nodes["server"] = {
            "id": now_nodes,
            "label": "<b>server</b>",
            "value": 1,
            "color": green,
        };
        server_nodes_to = new Set();
    }
    update_server_nodes_to(server_nodes_to, task_id);
    update_network(all_nodes, server_nodes_to, client_nodes_to);
});
socket.on("s_train_aggre_complete", function (res) {
    console.log("train_aggre_complete")
    let gep = res["gep"];
    let task_id = "train_aggre";
    if (all_nodes.hasOwnProperty(task_id)) {
        all_nodes[task_id].color = {
            "border": gray,
            "background": "#F8F8F8"
        };
        all_nodes[task_id].label = "<b>server</b>\n<i>train_aggre:</i><b>" + gep + "</b>\n<b>complete</b>";
    } else {
        now_nodes++;
        all_nodes[task_id] = {
            "id": now_nodes,
            "label": "<b>server</b>\n<i>train_aggre:</i><b>" + gep + "</b>\n<b>complete</b>",
            "value": 1,
            "color": {
                "border": gray,
                "background": "#F8F8F8"
            },
        };
    }
    if (!all_nodes.hasOwnProperty("server")) {
        now_nodes++;
        all_nodes["server"] = {
            "id": now_nodes,
            "label": "<b>server</b>",
            "value": 1,
            "color": green,
        };
        server_nodes_to = new Set();
    }
    update_server_nodes_to(server_nodes_to, task_id);
    update_network(all_nodes, server_nodes_to, client_nodes_to);
});
socket.on("c_eval", function (res) {
    console.log("eval")
    let sid = res["sid"];
    let gep = res["gep"];
    let task_id = "eval_" + sid;
    if (!all_nodes.hasOwnProperty(sid)) {
        now_nodes++;
        now_client++;
        let now_label = "<b>client_" + now_client + "</b>";
        all_nodes[sid] = {
            "id": now_nodes,
            "label": now_label,
            "value": 1,
            "color": green,
        };
    }
    let now_client_id = all_nodes[sid].label;
    if (all_nodes.hasOwnProperty(task_id)) {
        all_nodes[task_id].color = green;
        all_nodes[task_id].label = now_client_id + "\n<i>eval:</i><b>" + gep + "</b>";
    } else {
        now_nodes++;
        let now_client_id = all_nodes[sid].label;
        let now_label = now_client_id + "\n<i>eval:</i><b>" + gep + "</b>";
        all_nodes[task_id] = {
            "id": now_nodes,
            "label": now_label,
            "value": 1,
            "color": green,
        };
    }
    if (!all_nodes.hasOwnProperty("server")) {
        now_nodes++;
        all_nodes["server"] = {
            "id": now_nodes,
            "label": "<b>server</b>",
            "value": 1,
            "color": green,
        };
        server_nodes_to = new Set();
    }
    update_client_nodes_to(client_nodes_to, sid, "server")
    update_client_nodes_to(client_nodes_to, sid, task_id);
    update_network(all_nodes, server_nodes_to, client_nodes_to);
});
socket.on("c_eval_complete", function (res) {
    console.log("eval_complete")
    let sid = res["sid"];
    let gep = res["gep"];
    let task_id = "eval_" + sid;
    if (!all_nodes.hasOwnProperty(sid)) {
        now_nodes++;
        now_client++;
        let now_label = "<b>client_" + now_client + "</b>";
        all_nodes[sid] = {
            "id": now_nodes,
            "label": now_label,
            "value": 1,
            "color": green,
        };
    }
    let now_client_id = all_nodes[sid].label;
    if (all_nodes.hasOwnProperty(task_id)) {
        all_nodes[task_id].color = {
            "border": gray,
            "background": "#F8F8F8"
        };
        all_nodes[task_id].label = now_client_id + "\n<i>eval:</i><b>" + gep + "</b>\n<b>complete</b>";
    } else {
        now_nodes++;
        let now_label = now_client_id + "\n<i>eval:</i><b>" + gep + "</b>\n<b>complete</b>";
        all_nodes[task_id] = {
            "id": now_nodes,
            "label": now_label,
            "value": 1,
            "color": {
                "border": gray,
                "background": "#F8F8F8"
            },
        };
    }
    if (!all_nodes.hasOwnProperty("server")) {
        now_nodes++;
        all_nodes["server"] = {
            "id": now_nodes,
            "label": "<b>server</b>",
            "value": 1,
            "color": green,
        };
        server_nodes_to = new Set();
    }
    update_client_nodes_to(client_nodes_to, sid, "server")
    update_client_nodes_to(client_nodes_to, sid, task_id);
    update_network(all_nodes, server_nodes_to, client_nodes_to);
});
socket.on("s_eval_aggre", function (res) {
    console.log("eval_aggre")
    let gep = res["gep"];
    let task_id = "eval_aggre";
    if (all_nodes.hasOwnProperty(task_id)) {
        all_nodes[task_id].color = green;
        all_nodes[task_id].label = "<b>server</b>\n<i>eval_aggre:</i><b>" + gep + "</b>";
    } else {
        now_nodes++;
        all_nodes[task_id] = {
            "id": now_nodes,
            "label": "<b>server</b>\n<i>eval_aggre:</i><b>" + gep + "</b>",
            "value": 1,
            "color": green,
        }
    }
    if (!all_nodes.hasOwnProperty("server")) {
        now_nodes++;
        all_nodes["server"] = {
            "id": now_nodes,
            "label": "<b>server</b>",
            "value": 1,
            "color": green,
        };
        server_nodes_to = new Set();
    }
    update_server_nodes_to(server_nodes_to, task_id);
    update_network(all_nodes, server_nodes_to, client_nodes_to);
});
socket.on("s_eval_aggre_complete", function (res) {
    console.log("eval_aggre_complete")
    let gep = res["gep"];
    let task_id = "eval_aggre";
    if (all_nodes.hasOwnProperty(task_id)) {
        all_nodes[task_id].color = {
            "border": gray,
            "background": "#F8F8F8"
        };
        all_nodes[task_id].label = "<b>server</b>\n<i>eval_aggre:</i><b>" + gep + "</b>\n<b>complete</b>";
    } else {
        now_nodes++;
        all_nodes[task_id] = {
            "id": now_nodes,
            "label": "<b>server</b>\n<i>eval_aggre:</i><b>" + gep + "</b>\n<b>complete</b>",
            "value": 1,
            "color": {
                "border": gray,
                "background": "#F8F8F8"
            },
        }
    }
    if (!all_nodes.hasOwnProperty("server")) {
        now_nodes++;
        all_nodes["server"] = {
            "id": now_nodes,
            "label": "<b>server</b>",
            "value": 1,
            "color": green,
        };
        server_nodes_to = new Set();
    }
    update_server_nodes_to(server_nodes_to, task_id);
    update_network(all_nodes, server_nodes_to, client_nodes_to);
});
socket.on("s_summary", function () {
    console.log("summary")
    let task_id = "summary";
    if (all_nodes.hasOwnProperty(task_id)) {
        all_nodes[task_id].color = green;
        all_nodes[task_id].label = "<b>server</b>\n<i>summary</i>";
    } else {
        now_nodes++;
        all_nodes[task_id] = {
            "id": now_nodes,
            "label": "<b>server</b>\n<i>summary</i>",
            "value": 1,
            "color": green,
        }
    }
    if (!all_nodes.hasOwnProperty("server")) {
        now_nodes++;
        all_nodes["server"] = {
            "id": now_nodes,
            "label": "<b>server</b>",
            "value": 1,
            "color": green,
        };
        server_nodes_to = new Set();
    }
    update_server_nodes_to(server_nodes_to, task_id);
    update_network(all_nodes, server_nodes_to, client_nodes_to);
});
socket.on("s_summary_complete", function () {
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
            "value": 1,
            "color": {
                "border": gray,
                "background": "#F8F8F8"
            },
        }
    }
    if (!all_nodes.hasOwnProperty("server")) {
        now_nodes++;
        all_nodes["server"] = {
            "id": now_nodes,
            "label": "<b>server</b>",
            "value": 1,
            "color": green,
        };
        server_nodes_to = new Set();
    }
    update_server_nodes_to(server_nodes_to, task_id);
    update_network(all_nodes, server_nodes_to, client_nodes_to);
});
socket.on("c_fin", function (res) {
    console.log("client_fin")
    let sid = res["sid"];
    if (all_nodes.hasOwnProperty(sid)) {
        all_nodes[sid].color = {
            "border": gray,
            "background": "#F8F8F8"
        };
    }
    update_network(all_nodes, server_nodes_to, client_nodes_to);
});
socket.on("s_fin", function () {
    console.log("server_fin")
    if (all_nodes.hasOwnProperty("server")) {
        all_nodes["server"].color = {
            "border": gray,
            "background": "#F8F8F8"
        };
    }
    update_network(all_nodes, server_nodes_to, client_nodes_to);
});


// var nodes = new vis.DataSet(
//     [
//         {
//             id: 1,
//             label: 'Henry Hooper',
//             value: 1
//         },
//         {
//             id: 2,
//             label: '<b>Dr. Priya Gupta</b>\n<i>Cardiologist</i>',
//             color: {
//                 border: gray,
//                 background: '#F8F8F8'
//             },
//             borderWidth: 3,
//             value: 2
//         },
//         {
//             id: 3,
//             label: 'Lisinopril-GA tablet 10mg',
//             color: orange,
//             value: 3
//         },
//         {
//             id: 4,
//             label: 'Doxycycline',
//             value: 4,
//             title: "<b>Happy</b>",
//             color: green
//         },
//         {
//             id: 5,
//             label: 'Hydrochlorothiazide 25mg',
//             color: green,
//             value: 4,
//             title: "Happy"
//         },
//         {
//             id: 6,
//             label: 'Hydrochloroth 25mg',
//             color: red,
//             value: 2,
//             title: "Happy"
//         },
//         {
//             id: 7,
//             label: '<b>Dr. Mike Smith</b>\n<i>General Practicioner</i>',
//             color: {
//                 border: gray,
//                 background: '#F8F8F8'
//             },
//             value: 2,
//             borderWidth: 3
//         },
//         {
//             id: 8,
//             label: 'Lisinopril-GA tablet 10mg',
//             color: orange,
//             value: 3
//         },
//         {
//             id: 9,
//             label: 'Doxycycline',
//             value: 4,
//             title: "Happy",
//             color: green
//         },
//         {
//             id: 10,
//             label: 'Hydrochlorothiazide 25mg',
//             color: green,
//             value: 4,
//             title: "Happy"
//         },
//         {
//             id: 11,
//             label: 'Multi 25mg',
//             color: red,
//             value: 2,
//             title: "Happy"
//         }
//     ]);
//
// // create an array with edges
// var edges = new vis.DataSet([
//     {from: 1, to: 2},
//     {from: 2, to: 3},
//     {from: 2, to: 4},
//     {from: 2, to: 5},
//     {from: 2, to: 6},
//     {from: 1, to: 7},
//     {from: 7, to: 8},
//     {from: 7, to: 9},
//     {from: 7, to: 10},
//     {from: 7, to: 11}
// ]);
//
// // create a network
// var container = document.getElementById('mynetwork');
//
// // provide the data in the vis format
// var data = {
//     nodes: nodes,
//     edges: edges
// };


// initialize your network!
// var network = new vis.Network(container, data, options);