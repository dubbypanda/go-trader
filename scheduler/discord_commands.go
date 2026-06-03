package main

import (
	"fmt"

	"github.com/bwmarrin/discordgo"
)

// readOnlyCommandNames are usable in a guild or in DMs by anyone.
var readOnlyCommandNames = map[string]bool{
	"status":           true,
	"health":           true,
	"positions":        true,
	"pnl":              true,
	"leaderboard":      true,
	"circuit-breakers": true,
	"dead-strategies":  true,
	"correlation":      true,
	"logs":             true,
}

// opsCommandNames mutate state or run heavy work; restricted to the owner in a DM.
var opsCommandNames = map[string]bool{
	"restart":  true,
	"backtest": true,
}

// authorizeCommand decides whether invokerID may run command `name`. Read-only
// commands are always allowed. Ops commands require the invoker to be the owner
// AND the interaction to be a DM (guildID == ""). Returns (false, reason) on deny.
func authorizeCommand(name, invokerID, guildID, ownerID string) (bool, string) {
	if readOnlyCommandNames[name] {
		return true, ""
	}
	if opsCommandNames[name] {
		if ownerID == "" {
			return false, "owner is not configured; ops commands are disabled"
		}
		if invokerID != ownerID {
			return false, "not authorized — this command is owner-only"
		}
		if guildID != "" {
			return false, "this command is only available in a DM with the bot"
		}
		return true, ""
	}
	return false, fmt.Sprintf("unknown command: %s", name)
}

// interactionUserID extracts the invoking user's ID from either a guild
// (i.Member.User) or DM (i.User) interaction.
func interactionUserID(i *discordgo.InteractionCreate) string {
	if i.Member != nil && i.Member.User != nil {
		return i.Member.User.ID
	}
	if i.User != nil {
		return i.User.ID
	}
	return ""
}
