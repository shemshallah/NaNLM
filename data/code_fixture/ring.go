// SPDX-License-Identifier: MIT
package ring

import (
	"errors"
	"hash/fnv"
)

type Shard struct {
	ID      string
	Records int
	Hash    uint32
}

func HashID(id string) uint32 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(id))
	return h.Sum32()
}

func (s *Shard) Seal() error {
	if s.Records < 0 || s.ID == "" {
		return errors.New("invalid shard")
	}
	s.Hash = HashID(s.ID)
	return nil
}

func Assign(shard Shard, runners int) (int, error) {
	if runners <= 0 {
		return 0, errors.New("no runners")
	}
	if err := shard.Seal(); err != nil {
		return 0, err
	}
	return int(shard.Hash % uint32(runners)), nil
}
